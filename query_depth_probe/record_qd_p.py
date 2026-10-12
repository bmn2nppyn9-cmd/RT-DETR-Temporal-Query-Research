# -*- coding: utf-8 -*-
"""record_qd_p.py —— QD-P 阶段: 三层模型 Query-depth schedule 录制
================================================================
P0=300/300/300 (A0), P1=300/150/75, P2=300/200/100, P3=300/100/50,
策略: confidence / stability / confidence+stability (在线) / random (同预算控制) /
gt_oracle (仅上限; 掩码离线预计算, 模型内部无 GT)。

**真实执行 active subset**: 退出 query 冻结该层输出, 后续层只跑存活子集
(self-attention 只见存活 query), 最终合并回槽位统一 top-300。

每帧记录: 输出 + 每输出条 qidx/src/src_depth/inj_layer/exit_layer
+ 每 query 的 exit_ann (Nf,300) + 各层实际宽度 (Nf,3)
+ predict wall + decoder-only CUDA + 整体 CUDA 计时。

gt_oracle 每帧两次前向: 导出 pass (逐层正确性) + 掩码 pass (真实执行)。
协议: warmup 在全部探针关闭状态完成。

用法:
    python record_qd_p.py --limit-scenes 2          # smoke 全矩阵
    python record_qd_p.py --configs ...             # 全量 (确认后)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, r'C:\Users\TONY\ultralytics')
sys.path.insert(0, r'D:\lunwen\自研')
sys.path.insert(0, r'D:\lunwen\自研\query_prop_b0')
import ultralytics  # noqa: E402
assert ultralytics.__file__.lower().startswith(r'c:\users\tony\ultralytics\ultralytics'), \
    f'导入路径错误: {ultralytics.__file__}'
from eval_b0 import match_frame_tp  # noqa: E402
from eval_sap import load_gt  # noqa: E402

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
CKPT = r'D:\lunwen\自研\runs\detect\runs_rtdetr\r18_640_sub20_b4\weights\best.pt'
OUT = r'D:\lunwen\自研\query_depth_probe\stream_preds'

SCHEDULES = {'P0': [300, 300, 300], 'P1': [300, 150, 75], 'P2': [300, 200, 100], 'P3': [300, 100, 50]}
POLICIES = ['confidence', 'stability', 'confidence+stability', 'random', 'gt_oracle']
POL_TAG = {'confidence': 'conf', 'stability': 'stab', 'confidence+stability': 'cs',
           'random': 'rand', 'gt_oracle': 'oracle'}
IMG_W, IMG_H = 1920, 1200


def log(m):
    print(m, flush=True)


def load_bgr(path):
    with Image.open(path) as im:
        return np.array(im.convert('RGB'))[:, :, ::-1].copy()


def find_head(model):
    for mod in model.modules():
        if type(mod).__name__ == 'RTDETRDecoder' and hasattr(mod, 'decoder'):
            return mod
    return None


def cxcywh_to_xyxy(cxcywh):
    x = cxcywh[:, 0] * IMG_W
    y = cxcywh[:, 1] * IMG_H
    bw = cxcywh[:, 2] * IMG_W
    bh = cxcywh[:, 3] * IMG_H
    return np.stack([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2], -1).astype(np.float32)


def iou_pair(a, b):
    lt = np.maximum(a[:, :2], b[:, :2])
    rb = np.minimum(a[:, 2:], b[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, 0] * wh[:, 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa + bb - inter, 1e-9)


def random_masks(schedule, seed):
    """按调度生成随机退出掩码; 位置语义 = 当前存活子集 (升序槽位)"""
    rng = np.random.default_rng(seed)
    cur = np.arange(schedule[0])
    masks = {}
    for li in range(len(schedule) - 1):  # 每层 (除终层) 后一个退出点
        keep = rng.choice(cur, size=schedule[li + 1], replace=False)
        mm = np.ones(len(cur), bool)
        mm[np.isin(cur, keep)] = False
        masks[li] = torch.from_numpy(mm).unsqueeze(0).cuda()
        cur = keep
    return masks


def oracle_masks(schedule, gb, gc, ls, scores3):
    """GT oracle 掩码 (仅上限): 由导出 pass 的逐层状态算每 query 最早持续正确层。

    退出先排"该层起持续正确"的最终 TP (按 L3 分数降序), 不足再用非 TP 低分补齐。
    位置语义与 random_masks 一致 (升序槽位); 模型内部不接触任何 GT。
    """
    nL = len(ls)
    # 各层 (300,) box xyxy / cls
    layers = []
    for li in range(nL):
        b = cxcywh_to_xyxy(ls[li][1][0].cpu().numpy())
        c = np.argmax(ls[li][2][0].cpu().numpy(), -1)
        layers.append((b, c))
    # L3 最终 TP (类感知贪心, 分数降序)
    b3, c3 = layers[2]
    o = np.argsort(-scores3, kind='stable')
    tpf, det2gt = match_frame_tp(b3[o], scores3[o], c3[o], gb, gc, 0.5)
    gt_of = np.full(300, -1, np.int32)
    for d_ in np.nonzero(tpf)[0]:
        gt_of[o[d_]] = det2gt[d_]
    # 每 query 最早持续正确层 (0/1/2; 非 TP 为 6)
    d_gt = np.full(300, 6, np.int8)
    for q in np.nonzero(gt_of >= 0)[0]:
        g = gt_of[q]
        gbox = gb[g:g + 1]
        corr = np.zeros(nL, bool)
        for li in range(nL):
            corr[li] = layers[li][1][q] == gc[g] and iou_pair(layers[li][0][q:q + 1], gbox)[0] >= 0.5
        if corr[2]:
            for li in range(nL):
                if corr[li:].all():
                    d_gt[q] = li
                    break
    cur = np.arange(schedule[0])
    masks = {}
    for li in range(len(schedule) - 1):
        want_exit = len(cur) - schedule[li + 1]
        if want_exit <= 0:
            masks[li] = torch.zeros(1, len(cur), dtype=torch.bool).cuda()
            continue
        d = d_gt[cur]
        s3 = scores3[cur]
        # 优先级: 该层起持续正确的最终 TP 先退出 (按 L3 分数降序); 不足则非 TP 低分补齐
        cand_correct = cur[d == li]
        cand_fill = cur[d == 6]
        order_c = cand_correct[np.argsort(-s3[d == li], kind='stable')]
        order_f = cand_fill[np.argsort(s3[d == 6], kind='stable')]
        exit_ids = np.concatenate([order_c[:want_exit], order_f[:max(0, want_exit - len(order_c))]])
        mm = np.isin(cur, exit_ids)
        masks[li] = torch.from_numpy(mm).unsqueeze(0).cuda()
        cur = cur[~mm]
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stream-index', default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--images', default=os.path.join(DATA, 'images'))
    ap.add_argument('--ann', default=os.path.join(DATA, 'annotations_val.json'))
    ap.add_argument('--schedules', nargs='+', default=list(SCHEDULES))
    ap.add_argument('--policies', nargs='+', default=POLICIES)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--max-det', type=int, default=300)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--limit-scenes', type=int, default=0)
    ap.add_argument('--prefetch', type=int, default=8)
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.stream_index, encoding='utf-8') as f:
        idx = json.load(f)
    scenes = idx['scenes']
    if args.limit_scenes:
        scenes = scenes[:args.limit_scenes]
    n_frames = sum(s['n_frames'] for s in scenes)
    log(f'流式索引: {len(scenes)} 场景 / {n_frames} 帧')

    from ultralytics import RTDETR
    m = RTDETR(CKPT)
    head = find_head(m.model)
    if head is None:
        log('[错误] 找不到 RTDETRDecoder')
        sys.exit(1)
    head.decoder.eval_idx = 2
    head.set_temporal_enabled(False)
    head.set_query_overrides(encoder_queries=300, temporal_queries=0, output_topk=300, motion_alpha=0.0)
    head.set_exit_plan(None)

    # 协议: predictor warmup 在全部探针关闭状态完成
    warm_img = load_bgr(os.path.join(args.images, scenes[0]['frames'][0]['file']))
    _ = m.predict(warm_img, imgsz=args.imgsz, conf=args.conf, max_det=args.max_det,
                  device=0, verbose=False)[0]
    log('warmup 完成 (probe off)')

    gt_all = load_gt(args.ann)

    # 配置列表: P0 只有 A0 基线 (无退出); P1-P3 × 各策略
    # 基线标签用 a0p, 避免覆盖 QD-T 的 preds_a0.npz
    runs = [('a0p', None, None)]
    for sc_name in args.schedules:
        if sc_name == 'P0':
            continue
        for pol in args.policies:
            runs.append((f'p{sc_name[1]}_{POL_TAG[pol]}', SCHEDULES[sc_name], pol))

    for tag, sched, pol in runs:
        out_path = os.path.join(args.out, f'preds_{tag}.npz')
        if args.skip_existing and os.path.exists(out_path):
            log(f'\n[跳过] {out_path} 已存在')
            continue
        head.set_exit_plan(sched, pol) if sched else head.set_exit_plan(None)
        head.time_decoder_cuda = True
        head.time_model_cuda = True
        log('')
        log('=' * 60)
        log(f'录制 {tag} schedule={sched} policy={pol} -> {out_path}')
        log('=' * 60)

        scene_arr, fidx_arr, t_arr, lat_arr, dec_arr, mod_arr = [], [], [], [], [], []
        boxes, scores, clss, qidxs, srcs, sdeps, injs, exl = [], [], [], [], [], [], [], []
        ptr = [0]
        ann_arr, wid_arr = [], []
        t_start = time.time()
        done = 0
        src_mismatch = 0

        def predict_one(img):
            return m.predict(img, imgsz=args.imgsz, conf=args.conf, max_det=args.max_det,
                             device=0, verbose=False)[0]

        with ThreadPoolExecutor(max_workers=args.prefetch) as pool:
            for s in scenes:
                frs = s['frames']
                ts = [fr['t_ns'] for fr in frs]
                chunk = 32
                for base in range(0, len(frs), chunk):
                    part = frs[base:base + chunk]
                    files = [os.path.join(args.images, fr['file']) for fr in part]
                    imgs = list(pool.map(load_bgr, files))
                    for k, im in enumerate(imgs):
                        t0 = time.perf_counter()
                        if pol == 'gt_oracle':
                            # 导出 pass (无退出, 逐层状态) — 本帧计时只计掩码 pass
                            head.set_exit_plan(None)
                            head.return_all_layers = True
                            _ = predict_one(im)
                            ls = head._last_layer_states
                            head.return_all_layers = False
                            head.set_exit_plan(sched, pol)
                            gb, gc = gt_all[part[k]['file']]
                            # scores3 = L3 最大分数 (300,)
                            scores3 = (1.0 / (1.0 + np.exp(-ls[2][2][0].cpu().numpy()))).max(-1)
                            head._exit_masks = oracle_masks(sched, gb, gc, ls, scores3)
                            t0 = time.perf_counter()
                            r = predict_one(im)
                        else:
                            if pol == 'random':
                                # 确定性种子 (Python hash 随机化不可用): scene_id*1e6 + frame_idx
                                head._exit_masks = random_masks(sched, int(s['scene_id']) * 1000000 + base + k)
                            r = predict_one(im)
                        lat = (time.perf_counter() - t0) * 1000.0

                        b = r.boxes
                        n = 0 if b is None else len(b)
                        meta = head._last_topk_meta
                        keep = (meta['scores'][0] > args.conf).nonzero(as_tuple=True)[0]
                        keep = keep[:args.max_det]
                        src = meta['source'][0][keep].cpu().numpy().astype(np.int8)
                        if len(src) != n:
                            src_mismatch += 1
                            src = np.full(n, -1, np.int8)
                        qidx = meta['query_idx'][0][keep].cpu().numpy().astype(np.int16)
                        sdep = meta['source_depth'][0][keep].cpu().numpy().astype(np.int8)
                        inj = meta['injection_layer'][0][keep].cpu().numpy().astype(np.int8)
                        ex = meta['exit_layer'][0][keep].cpu().numpy().astype(np.int8)
                        if n:
                            boxes.append(b.xyxy.cpu().numpy().astype(np.float32))
                            scores.append(b.conf.cpu().numpy().astype(np.float32))
                            clss.append(b.cls.cpu().numpy().astype(np.uint8))
                            qidxs.append(qidx)
                            srcs.append(src)
                            sdeps.append(sdep)
                            injs.append(inj)
                            exl.append(ex)
                        ptr.append(ptr[-1] + n)

                        ann = head._last_exit_ann[0].cpu().numpy().astype(np.int8) if sched else np.full(300, -1, np.int8)
                        wid = np.array(head._last_exit_widths, np.int16) if sched else np.array([300, 300, 300], np.int16)
                        ann_arr.append(ann)
                        wid_arr.append(wid)
                        scene_arr.append(s['scene_id'])
                        fidx_arr.append(base + k)
                        t_arr.append(ts[base + k])
                        lat_arr.append(lat)
                        dec_arr.append(head._last_decoder_ms)
                        mod_arr.append(head._last_model_ms)
                        done += 1
                    el = time.time() - t_start
                    if done % 512 == 0 or done == n_frames:
                        log(f'  {done}/{n_frames}  {el:.0f}s  '
                            f'{done / max(el, 1e-9):.1f} 帧/秒  近帧耗时 {lat:.1f}ms')

        np.savez_compressed(
            out_path,
            scene=np.array(scene_arr, np.int16),
            frame_idx=np.array(fidx_arr, np.int16),
            t_ns=np.array(t_arr, np.int64),
            compute_ms=np.array(lat_arr, np.float32),
            dec_ms=np.array(dec_arr, np.float32),
            model_ms=np.array(mod_arr, np.float32),
            box=(np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32)),
            score=(np.concatenate(scores) if scores else np.zeros(0, np.float32)),
            cls=(np.concatenate(clss) if clss else np.zeros(0, np.uint8)),
            qidx=(np.concatenate(qidxs) if qidxs else np.zeros(0, np.int16)),
            src=(np.concatenate(srcs) if srcs else np.zeros(0, np.int8)),
            src_depth=(np.concatenate(sdeps) if sdeps else np.zeros(0, np.int8)),
            inj_layer=(np.concatenate(injs) if injs else np.zeros(0, np.int8)),
            exit_layer=(np.concatenate(exl) if exl else np.zeros(0, np.int8)),
            ptr=np.array(ptr, np.int32),
            exit_ann=np.array(ann_arr, np.int8),
            widths=np.array(wid_arr, np.int16),
            config=np.str_(tag),
        )
        med = float(np.median(lat_arr))
        p95 = float(np.percentile(lat_arr, 95))
        log(f'  完成: {done} 帧, 推理 中位 {med:.2f}ms / P95 {p95:.2f}ms, '
            f'decoder-only 中位 {float(np.median(dec_arr)):.3f}ms, 整体 中位 {float(np.median(mod_arr)):.2f}ms, '
            f'src 对齐失败 {src_mismatch} 帧')
        log(f'  已写入 {out_path}')

    head.set_exit_plan(None)
    head.time_decoder_cuda = False
    head.time_model_cuda = False
    log('')
    log('全部配置录制完成。')


if __name__ == '__main__':
    main()
