# -*- coding: utf-8 -*-
"""
probe_encoder_recall.py —— 诊断: encoder 提案 vs 传播锚点的质量对比
=====================================================================
要回答的问题:
    RT-DETR 的 decoder 初始 query 来自 encoder 的 top-300 提案(enc_bboxes)。
    这个"起点"的质量, 跟"上一帧传播过来的锚点"相比谁更好?
    这决定路径B的注入方式是【替换 / 混合 / 不用】。

关键坐标变换(最容易错的地方):
    RT-DETR 的 predict 用 LetterBox(imgsz, auto=False, scale_fill=True)
    —— 即【直接拉伸, 不保持宽高比, 无 padding】:
        1920x1200 -> 640x640   横向 x/3.0 , 纵向 y/1.875
    若误用"等比缩放+居中黑边"的公式反投影, 坐标会整体偏移, recall 归零。

四个探针(同一批帧, 帧序号>=2):
    encoder : encoder top-300 提案(enc_bboxes)          —— 现状的起点
    copy    : 前一帧的检测框                             —— 传播(零阶)
    extrap  : 前一帧 + 0.5*(前帧-前前帧)                 —— 传播(阻尼外推)
    decoder : decoder 最终输出                          —— 上界参照

指标: recall@IoU(0.3/0.5/0.7) + 每帧框数。
      用 recall 而非 mAP, 因为起点好坏的本质是"覆盖了多少 GT", 与分数排序无关。

用法:
    python probe_encoder_recall.py --limit 300      # 先小样本验证坐标变换
    python probe_encoder_recall.py                  # 全量
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from eval_sap import iou_matrix, load_gt  # noqa: E402

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'


def log(m):
    print(m, flush=True)


def recall_at(dets_xyxy, gts_xyxy, thr):
    """dets 对 gts 的召回: 有多少 GT 被至少一个 det 覆盖(IoU>=thr)"""
    if len(gts_xyxy) == 0:
        return 0, 0
    if len(dets_xyxy) == 0:
        return 0, len(gts_xyxy)
    iou = iou_matrix(dets_xyxy, gts_xyxy)          # (D, G)
    hit = int((iou.max(axis=0) >= thr).sum())
    return hit, len(gts_xyxy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',
                    default=r'runs\detect\runs_rtdetr\r18_640_sub20_b4\weights\best.pt')
    ap.add_argument('--preds', default='stream_preds/preds_d6.npz',
                    help='用于生成 copy/extrap 探针的录制预测')
    ap.add_argument('--ann', default=os.path.join(DATA, 'annotations_val.json'))
    ap.add_argument('--stream-index', default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--images', default=os.path.join(DATA, 'images'))
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 帧(验证坐标变换用)')
    ap.add_argument('--out', default='out_probe_encoder.json')
    args = ap.parse_args()

    with open(args.stream_index, encoding='utf-8') as f:
        idx = json.load(f)
    gt = load_gt(args.ann)

    # 源预测(d6) 查表
    z = np.load(args.preds, allow_pickle=False)
    rec = {k: z[k] for k in z.files}
    per_frame = {}
    for i in range(len(rec['scene'])):
        s, e = int(rec['ptr'][i]), int(rec['ptr'][i + 1])
        per_frame[(int(rec['scene'][i]), int(rec['frame_idx'][i]))] = (
            rec['box'][s:e], rec['score'][s:e], rec['cls'][s:e])

    # 评测帧: 每场景 idx>=2
    targets = []
    for sc in idx['scenes']:
        sid = sc['scene_id']
        for fr in sc['frames']:
            if fr['idx'] >= 2 and (sid, fr['idx']) in per_frame:
                targets.append((sid, fr['idx'], fr['file']))
    if args.limit:
        targets = targets[:args.limit]
    log(f'评测帧: {len(targets)}')

    # --- 坐标变换: scale_fill=True => 直接拉伸 ---
    W, H = 1920.0, 1200.0
    sx, sy = W / args.imgsz, H / args.imgsz      # 640 -> 原图 的缩放
    log(f'坐标变换(scale_fill 拉伸): x*{sx:.3f}, y*{sy:.3f}')

    from ultralytics import RTDETR
    m = RTDETR(args.model)
    net = m.model.cuda().eval()
    # 找 RTDETRDecoder
    dec = None
    for mod in net.modules():
        if type(mod).__name__ == 'RTDETRDecoder':
            dec = mod
            break
    if dec is None:
        log('[错误] 找不到 RTDETRDecoder')
        sys.exit(1)
    log(f'RTDETRDecoder: nl={dec.nl} nq={dec.num_queries}')

    accum = {k: {0.3: [0, 0], 0.5: [0, 0], 0.7: [0, 0]}
             for k in ('encoder', 'copy', 'extrap', 'decoder')}
    n_box = {k: 0 for k in accum}
    t0 = time.time()

    for n, (sid, fi, fn) in enumerate(targets):
        img_path = os.path.join(args.images, fn)
        with Image.open(img_path) as im:
            rgb = im.convert('RGB')
            # scale_fill=True -> 直接拉伸, 不保持宽高比
            rgb = rgb.resize((args.imgsz, args.imgsz), Image.BILINEAR)
            arr = np.array(rgb)                       # (640,640,3) RGB uint8
        # 网络张量约定为 RGB(见 predictor.preprocess: im[..., ::-1] 是 BGR->RGB)。
        # 这里手动前向, 不再经过 predictor, 所以直接喂 RGB -> permute -> 归一化。
        t = torch.from_numpy(np.ascontiguousarray(arr))
        t = t.permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0

        with torch.no_grad():
            y = net(t)
        _y, (dec_bboxes, dec_scores, enc_bboxes, enc_scores, _dn) = y
        # 归一化坐标 -> 原图像素
        eb = enc_bboxes[0].float().cpu().numpy()          # (300,4) cxcywh in [0,1]
        db = dec_bboxes[-1][0].float().cpu().numpy()

        def to_xyxy(b):
            cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            out = np.stack([(cx - w / 2) * W, (cy - h / 2) * H,
                            (cx + w / 2) * W, (cy + h / 2) * H], 1)
            return out.astype(np.float32)

        probes = {'encoder': to_xyxy(eb), 'decoder': to_xyxy(db)}

        # 传播类探针: 用录制预测(原图像素坐标)
        b1, _, _ = per_frame[(sid, fi - 1)]
        b2, _, _ = per_frame[(sid, fi - 2)]
        probes['copy'] = b1
        if len(b1) and len(b2):
            iou = iou_matrix(b1, b2)
            best = iou.argmax(axis=1)
            ok = iou[np.arange(len(b1)), best] >= 0.3
            v = np.zeros_like(b1)
            v[ok] = b1[ok] - b2[best[ok]]
            probes['extrap'] = b1 + args.alpha * v
        else:
            probes['extrap'] = b1

        gb, _gc = gt[fn]
        for k, d in probes.items():
            n_box[k] += len(d)
            for thr in (0.3, 0.5, 0.7):
                h, t_ = recall_at(d, gb, thr)
                accum[k][thr][0] += h
                accum[k][thr][1] += t_

        if (n + 1) % 100 == 0 or n + 1 == len(targets):
            el = time.time() - t0
            log(f'  {n+1}/{len(targets)}  {el:.0f}s  {1000*el/(n+1):.0f}ms/帧')

    # --- sanity check: decoder 探针应接近模型自身 recall ---
    log('')
    log('=' * 74)
    log('指标: recall@IoU (被至少一个框覆盖的 GT 占比)')
    log(f"{'探针':<12}{'框数/帧':>9}{'R@0.3':>10}{'R@0.5':>10}{'R@0.7':>10}")
    for k in ('encoder', 'copy', 'extrap', 'decoder'):
        a = accum[k]
        row = f"{k:<12}{n_box[k]/max(len(targets),1):>9.1f}"
        for thr in (0.3, 0.5, 0.7):
            h, t_ = a[thr]
            row += f"{100.0*h/max(t_,1):>9.1f}%"
        log(row)
    log('=' * 74)
    log('注: decoder 行应与你之前测的 model recall@50≈80% 接近 —— 若偏离过大,')
    log('    说明坐标变换或预处理与 predict 路径不一致, 结论不可用。')

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(dict(n_frames=len(targets), alpha=args.alpha,
                       recall={k: {str(thr): (100.0 * v[0] / max(v[1], 1))
                                   for thr, v in accum[k].items()}
                               for k in accum},
                       boxes_per_frame={k: n_box[k] / max(len(targets), 1)
                                        for k in n_box}),
                  f, ensure_ascii=False, indent=2)
    log(f'已写入 {args.out}')


if __name__ == '__main__':
    main()
