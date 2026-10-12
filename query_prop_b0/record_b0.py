# -*- coding: utf-8 -*-
"""record_b0.py —— B0 探针四组对照的流式推理录制
=====================================================
沿用 record_stream_predictions.py 的口径(只计时 predict、conf=0.001、max_det=300),
按 stream_index 场景顺序逐帧推进, 每个场景边界 reset temporal state。

四组配置:
    A0: depth=3, 300 current encoder queries, 0 temporal   (标准三层基线)
    A1: depth=3, 375 current encoder queries, 0 temporal   (排除"只是查询变多")
    B0: depth=3, 300 current + 75 previous-frame hidden queries
    U0: depth=6, 300 current encoder queries, 0 temporal   (六层上界)

输出(每组一个 npz, 字段与 record_stream_predictions 兼容并扩展):
    scene / frame_idx / t_ns / compute_ms / box / score / cls / ptr
    src      (Nd,)   int8   每个输出的 query 来源: 0=current, 1=temporal (非 B0 全 0)
    nq_dec   (Nf,)   int16  该帧 decoder 实际输入 query 数 (300 或 375)

用法:
    python record_b0.py --configs A0 B0 --limit-scenes 2   # smoke
    python record_b0.py --configs A0 A1 B0 U0              # 全量
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

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'

CONFIGS = {
    'A0': dict(eval_idx=2, enc=300, temporal=False, ntmp=0),
    'A1': dict(eval_idx=2, enc=375, temporal=False, ntmp=0),
    'B0': dict(eval_idx=2, enc=300, temporal=True, ntmp=75),
    'U0': dict(eval_idx=5, enc=300, temporal=False, ntmp=0),
}


def log(m):
    print(m, flush=True)


def load_bgr(path):
    with Image.open(path) as im:
        return np.array(im.convert('RGB'))[:, :, ::-1].copy()


def find_head(model):
    """定位 RTDETRDecoder(head 模块本身, 不是它的 .decoder)"""
    for mod in model.modules():
        if type(mod).__name__ == 'RTDETRDecoder' and hasattr(mod, 'decoder'):
            return mod
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',
                    default=r'D:\lunwen\自研\runs\detect\runs_rtdetr\r18_640_sub20_b4\weights\best.pt')
    ap.add_argument('--stream-index', default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--images', default=os.path.join(DATA, 'images'))
    ap.add_argument('--configs', nargs='+', default=['A0', 'A1', 'B0', 'U0'])
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--max-det', type=int, default=300)
    ap.add_argument('--motion-alpha', type=float, default=0.0, help='B0 运动外推权重, 首轮为 0')
    ap.add_argument('--out', default=r'D:\lunwen\自研\query_prop_b0\stream_preds')
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
    log(f'加载模型 {args.model}')
    m = RTDETR(args.model)
    head = find_head(m.model)
    if head is None:
        log('[错误] 找不到 RTDETRDecoder')
        sys.exit(1)
    log(f'head: num_queries={head.num_queries}  decoder.eval_idx={head.decoder.eval_idx}')

    for cfg_name in args.configs:
        cfg = CONFIGS[cfg_name]
        tag = cfg_name.lower() if args.motion_alpha == 0 else f'{cfg_name.lower()}a{args.motion_alpha}'
        out_path = os.path.join(args.out, f'preds_{tag}.npz')
        if args.skip_existing and os.path.exists(out_path):
            log(f'\n[跳过] {out_path} 已存在')
            continue

        # 配置应用: 先彻底关闭并清空, 再按组设置
        head.set_temporal_enabled(False)
        head.set_query_overrides(encoder_queries=cfg['enc'], temporal_queries=cfg['ntmp'],
                                 output_topk=300, motion_alpha=args.motion_alpha)
        if cfg['temporal']:
            head.set_temporal_enabled(True)
        head.decoder.eval_idx = cfg['eval_idx']
        log('')
        log('=' * 60)
        log(f'录制 {cfg_name} -> {out_path}')
        log(f'  {head.temporal_status()}')
        log('=' * 60)

        scene_arr, fidx_arr, t_arr, lat_arr, nq_arr = [], [], [], [], []
        boxes, scores, clss, srcs, ptr = [], [], [], [], [0]
        t_start = time.time()
        done = 0
        src_mismatch = 0

        with ThreadPoolExecutor(max_workers=args.prefetch) as pool:
            for s in scenes:
                # 场景边界: 任何历史状态不得跨场景
                head.reset_temporal_state()
                files = [os.path.join(args.images, fr['file']) for fr in s['frames']]
                ts = [fr['t_ns'] for fr in s['frames']]
                chunk = 32
                for base in range(0, len(files), chunk):
                    part = files[base:base + chunk]
                    imgs = list(pool.map(load_bgr, part))
                    for k, im in enumerate(imgs):
                        t0 = time.perf_counter()
                        r = m.predict(im, imgsz=args.imgsz, conf=args.conf,
                                      max_det=args.max_det, device=0, verbose=False)[0]
                        lat = (time.perf_counter() - t0) * 1000.0

                        b = r.boxes
                        n = 0 if b is None else len(b)
                        # query 来源: 复现 predictor 的 conf>thr + max_det 过滤, 与输出逐一对齐
                        meta = head._last_topk_meta
                        keep = (meta['scores'][0] > args.conf).nonzero(as_tuple=True)[0]
                        keep = keep[:args.max_det]
                        src = meta['source'][0][keep].cpu().numpy().astype(np.int8)
                        if len(src) != n:
                            src_mismatch += 1
                            src = np.full(n, -1, np.int8)
                        if n:
                            boxes.append(b.xyxy.cpu().numpy().astype(np.float32))
                            scores.append(b.conf.cpu().numpy().astype(np.float32))
                            clss.append(b.cls.cpu().numpy().astype(np.uint8))
                            srcs.append(src)
                        ptr.append(ptr[-1] + n)

                        scene_arr.append(s['scene_id'])
                        fidx_arr.append(base + k)
                        t_arr.append(ts[base + k])
                        lat_arr.append(lat)
                        nq_arr.append(head._last_decoder_query_count)
                        done += 1
                    el = time.time() - t_start
                    if done % 512 == 0 or done == n_frames:
                        log(f'  {done}/{n_frames}  {el:.0f}s  '
                            f'{done/max(el,1e-9):.1f} 帧/秒  近帧耗时 {lat:.1f}ms')

        box_all = (np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32))
        np.savez_compressed(
            out_path,
            scene=np.array(scene_arr, np.int16),
            frame_idx=np.array(fidx_arr, np.int16),
            t_ns=np.array(t_arr, np.int64),
            compute_ms=np.array(lat_arr, np.float32),
            box=box_all,
            score=(np.concatenate(scores) if scores else np.zeros(0, np.float32)),
            cls=(np.concatenate(clss) if clss else np.zeros(0, np.uint8)),
            src=(np.concatenate(srcs) if srcs else np.zeros(0, np.int8)),
            ptr=np.array(ptr, np.int32),
            nq_dec=np.array(nq_arr, np.int16),
            config=np.str_(cfg_name),
            motion_alpha=np.float32(args.motion_alpha),
        )
        med = float(np.median(lat_arr))
        p95 = float(np.percentile(lat_arr, 95))
        log(f'  完成: {done} 帧, {len(box_all)} 个预测, '
            f'推理耗时 中位 {med:.2f}ms / P95 {p95:.2f}ms, src 对齐失败 {src_mismatch} 帧')
        log(f'  已写入 {out_path}')

    # 收尾: 恢复默认, 不留状态
    head.set_temporal_enabled(False)
    head.set_query_overrides(encoder_queries=300, temporal_queries=0, output_topk=300)
    log('')
    log('全部配置录制完成。')


if __name__ == '__main__':
    main()
