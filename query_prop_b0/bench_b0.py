# -*- coding: utf-8 -*-
"""bench_b0.py —— 四组配置的重复稳定计时 (每配置 3+ 次)
=====================================================
在固定帧子集(场景 0 前 N 帧)上, 每配置重复 repeat 次推理,
报告每次的 median / P95。B0 每次重复前 reset state, 重复内按流式连续推进。

用法:
    python bench_b0.py --configs A0 A1 B0 U0 --frames 200 --repeat 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

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
    for mod in model.modules():
        if type(mod).__name__ == 'RTDETRDecoder' and hasattr(mod, 'decoder'):
            return mod
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',
                    default=r'D:\lunwen\自研\runs\detect\runs_rtdetr\r18_640_sub20_b4\weights\best.pt')
    ap.add_argument('--configs', nargs='+', default=['A0', 'A1', 'B0', 'U0'])
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--out', default=r'D:\lunwen\自研\query_prop_b0\logs\bench_b0.json')
    args = ap.parse_args()

    with open(os.path.join(DATA, 'stream_index.json'), encoding='utf-8') as f:
        idx = json.load(f)
    files = [os.path.join(DATA, 'images', fr['file'])
             for fr in idx['scenes'][0]['frames'][:args.frames]]
    imgs = [load_bgr(p) for p in files]
    log(f'计时帧集: 场景0 前 {len(imgs)} 帧, 每配置 {args.repeat} 次重复 + {args.warmup} 预热')

    from ultralytics import RTDETR
    m = RTDETR(args.model)
    head = find_head(m.model)

    results = {}
    for cfg_name in args.configs:
        cfg = CONFIGS[cfg_name]
        head.set_temporal_enabled(False)
        head.set_query_overrides(encoder_queries=cfg['enc'], temporal_queries=cfg['ntmp'],
                                 output_topk=300, motion_alpha=0.0)
        if cfg['temporal']:
            head.set_temporal_enabled(True)
        head.decoder.eval_idx = cfg['eval_idx']

        # 预热
        head.reset_temporal_state()
        for im in imgs[:args.warmup]:
            m.predict(im, imgsz=args.imgsz, conf=0.001, max_det=300, device=0, verbose=False)

        runs = []
        for rep in range(args.repeat):
            head.reset_temporal_state()
            lat = []
            for im in imgs:
                t0 = time.perf_counter()
                m.predict(im, imgsz=args.imgsz, conf=0.001, max_det=300, device=0, verbose=False)
                lat.append((time.perf_counter() - t0) * 1000.0)
            lat = np.asarray(lat)
            runs.append(dict(median=float(np.median(lat)), p95=float(np.percentile(lat, 95)),
                             mean=float(lat.mean())))
            log(f'{cfg_name} 第{rep + 1}次: median {runs[-1]["median"]:.2f}ms  '
                f'P95 {runs[-1]["p95"]:.2f}ms')
        meds = [r['median'] for r in runs]
        p95s = [r['p95'] for r in runs]
        results[cfg_name] = dict(runs=runs, median_of_medians=float(np.median(meds)),
                                 median_of_p95=float(np.median(p95s)))

    head.set_temporal_enabled(False)
    head.set_query_overrides(encoder_queries=300, temporal_queries=0, output_topk=300)

    summary = {c: dict(median=v['median_of_medians'], p95=v['median_of_p95'])
               for c, v in results.items()}
    if 'A0' in summary and 'B0' in summary:
        summary['_b0_extra_ms_vs_a0'] = summary['B0']['median'] - summary['A0']['median']
    if 'U0' in summary and 'B0' in summary:
        summary['_b0_time_saved_pct_vs_u0'] = 100 * (1 - summary['B0']['median'] / summary['U0']['median'])
    results['_summary'] = summary
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f'已写入 {args.out}')


if __name__ == '__main__':
    main()
