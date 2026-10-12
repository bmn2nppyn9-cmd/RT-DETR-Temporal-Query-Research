# -*- coding: utf-8 -*-
"""record_qd_o.py —— QD-O 阶段: 六层模型逐 Query 中间层导出录制
=================================================================
U0 (eval_idx=5, 300 current, 无 temporal) 打开 return_all_layers 导出,
逐帧保存每层每 Query 的:

    l_box   (Nf, 6, 300, 4)  float32  归一化 sigmoid cxcywh (层 0..5)
    l_logit (Nf, 6, 300, 8)  float32  分类 logits (未过 sigmoid)
    l_ref   (Nf, 6, 300, 4)  float32  每层输入 reference (sigmoid cxcywh)
    det_qidx(Nf, 300)        int16    最终 top-300 各行对应的 query 槽位 (meta)

Query index 各层同槽位, 不经过 postprocess 展平追踪。
可选 --save-hidden-topk K: 额外保存 L6 分数 top-K query 的 hidden (Nf, 6, K, 256) float16。
hidden 只用于验证导出机制; 全量默认关闭 (15062 帧 × 300 槽 × 6 层 × 256 维 f16 ≈ 13.9GB)。

协议: predictor warmup 在 temporal 关闭状态完成, 之后配置导出并开始录制。

用法:
    python record_qd_o.py --limit-scenes 2 --save-hidden-topk 75   # smoke
    python record_qd_o.py                                          # 全量
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, r'C:\Users\TONY\ultralytics')
import ultralytics  # noqa: E402
assert ultralytics.__file__.lower().startswith(r'c:\users\tony\ultralytics\ultralytics'), \
    f'导入路径错误: {ultralytics.__file__}'

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
CKPT = r'D:\lunwen\自研\runs\detect\runs_rtdetr\r18_640_sub20_b4\weights\best.pt'
OUT = r'D:\lunwen\自研\query_depth_probe\stream_preds'


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stream-index', default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--images', default=os.path.join(DATA, 'images'))
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--max-det', type=int, default=300)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--tag', default='qd_o_u0', help='输出文件名 preds_<tag>.npz')
    ap.add_argument('--limit-scenes', type=int, default=0)
    ap.add_argument('--save-hidden-topk', type=int, default=0,
                    help='额外保存 L6 分数 top-K query 的 hidden (f16); 0 关闭')
    ap.add_argument('--prefetch', type=int, default=8)
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f'preds_{args.tag}.npz')
    if args.skip_existing and os.path.exists(out_path):
        log(f'[跳过] {out_path} 已存在')
        return

    with open(args.stream_index, encoding='utf-8') as f:
        idx = json.load(f)
    scenes = idx['scenes']
    if args.limit_scenes:
        scenes = scenes[:args.limit_scenes]
    n_frames = sum(s['n_frames'] for s in scenes)
    log(f'流式索引: {len(scenes)} 场景 / {n_frames} 帧 -> {out_path}')

    from ultralytics import RTDETR
    m = RTDETR(CKPT)
    head = find_head(m.model)
    if head is None:
        log('[错误] 找不到 RTDETRDecoder')
        sys.exit(1)
    head.decoder.eval_idx = 5  # U0: 六层
    head.set_temporal_enabled(False)
    head.set_query_overrides(encoder_queries=300, temporal_queries=0, output_topk=300, motion_alpha=0.0)

    # 协议: predictor warmup 在 temporal 关闭 + 导出关闭状态完成
    warm_img = load_bgr(os.path.join(args.images, scenes[0]['frames'][0]['file']))
    _ = m.predict(warm_img, imgsz=args.imgsz, conf=args.conf, max_det=args.max_det,
                  device=0, verbose=False)[0]
    log('warmup 完成 (temporal off / export off)')

    head.return_all_layers = True
    head.reset_temporal_state()

    n_hid = args.save_hidden_topk
    l_box = np.empty((n_frames, 6, 300, 4), np.float32)
    l_logit = np.empty((n_frames, 6, 300, 8), np.float32)
    l_ref = np.empty((n_frames, 6, 300, 4), np.float32)
    det_qidx = np.zeros((n_frames, 300), np.int16)
    det_cls = np.zeros((n_frames, 300), np.int16)
    hid = np.empty((n_frames, 6, n_hid, 256), np.float16) if n_hid else None
    scene_arr = np.zeros(n_frames, np.int16)
    fidx_arr = np.zeros(n_frames, np.int16)
    t_arr = np.zeros(n_frames, np.int64)
    lat_arr = np.zeros(n_frames, np.float32)

    t_start = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.prefetch) as pool:
        for s in scenes:
            files = [os.path.join(args.images, fr['file']) for fr in s['frames']]
            ts = [fr['t_ns'] for fr in s['frames']]
            chunk = 32
            for base in range(0, len(files), chunk):
                part = files[base:base + chunk]
                imgs = list(pool.map(load_bgr, part))
                for k, im in enumerate(imgs):
                    gi = done
                    t0 = time.perf_counter()
                    r = m.predict(im, imgsz=args.imgsz, conf=args.conf, max_det=args.max_det,
                                  device=0, verbose=False)[0]
                    lat_arr[gi] = (time.perf_counter() - t0) * 1000.0
                    ls = head._last_layer_states
                    assert ls is not None and sorted(ls.keys()) == [0, 1, 2, 3, 4, 5], '导出结构异常'
                    for li in range(6):
                        _, b, lg, rf = ls[li]
                        l_box[gi, li] = b[0].cpu().numpy()
                        l_logit[gi, li] = lg[0].cpu().numpy()
                        l_ref[gi, li] = rf[0].cpu().numpy()
                    if n_hid:
                        # hidden 按 L6 分数 top-K 保存, 仅用于验证导出机制
                        sc6 = 1.0 / (1.0 + np.exp(-l_logit[gi, 5]))
                        top = np.argsort(-sc6.max(-1))[:n_hid]
                        for li in range(6):
                            hid[gi, li] = ls[li][0][0, top].cpu().numpy().astype(np.float16)
                    meta = head._last_topk_meta
                    det_qidx[gi] = meta['query_idx'][0].cpu().numpy().astype(np.int16)
                    det_cls[gi] = meta['cls'][0].cpu().numpy().astype(np.int16)
                    scene_arr[gi] = s['scene_id']
                    fidx_arr[gi] = base + k
                    t_arr[gi] = ts[base + k]
                    done += 1
                el = time.time() - t_start
                if done % 512 == 0 or done == n_frames:
                    log(f'  {done}/{n_frames}  {el:.0f}s  {done / max(el, 1e-9):.1f} 帧/秒  近帧耗时 {lat_arr[done-1]:.1f}ms')

    head.return_all_layers = False
    head.set_temporal_enabled(False)

    payload = dict(
        scene=scene_arr, frame_idx=fidx_arr, t_ns=t_arr, compute_ms=lat_arr,
        l_box=l_box, l_logit=l_logit, l_ref=l_ref, det_qidx=det_qidx, det_cls=det_cls,
        config=np.str_('U0_6L_export'),
    )
    if hid is not None:
        payload['l_hidden_topk'] = hid
        payload['hidden_topk'] = np.int16(n_hid)
    np.savez_compressed(out_path, **payload)
    med = float(np.median(lat_arr))
    p95 = float(np.percentile(lat_arr, 95))
    log(f'  完成: {done} 帧, 导出开启下推理耗时 中位 {med:.2f}ms / P95 {p95:.2f}ms (含导出开销, 不用于计时声明)')
    log(f'  已写入 {out_path}')


if __name__ == '__main__':
    main()
