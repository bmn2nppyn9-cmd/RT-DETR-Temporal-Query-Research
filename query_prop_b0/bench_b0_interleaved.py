# -*- coding: utf-8 -*-
"""bench_b0_interleaved.py —— 轮转交错计时, 消除热漂移偏袒
每轮依次跑 A0/A1/B0/U0 各 N 帧, 共 R 轮; 同一轮内四组共享热状态。
"""
from __future__ import annotations

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
    n_frames, n_rounds, warmup = 150, 5, 30
    with open(os.path.join(DATA, 'stream_index.json'), encoding='utf-8') as f:
        idx = json.load(f)
    files = [os.path.join(DATA, 'images', fr['file'])
             for fr in idx['scenes'][0]['frames'][:n_frames + warmup]]
    imgs = [load_bgr(p) for p in files]
    log(f'交错计时: {n_rounds} 轮 x 4 配置 x {n_frames} 帧 (+{warmup} 预热)')

    from ultralytics import RTDETR
    m = RTDETR(r'D:\lunwen\自研\runs\detect\runs_rtdetr\r18_640_sub20_b4\weights\best.pt')
    head = find_head(m.model)

    def apply(cfg):
        head.set_temporal_enabled(False)
        head.set_query_overrides(encoder_queries=cfg['enc'], temporal_queries=cfg['ntmp'],
                                 output_topk=300, motion_alpha=0.0)
        if cfg['temporal']:
            head.set_temporal_enabled(True)
        head.decoder.eval_idx = cfg['eval_idx']
        head.reset_temporal_state()

    # 预热(混合四组)
    for c in CONFIGS.values():
        apply(c)
        for im in imgs[:warmup // 4 + 1]:
            m.predict(im, imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)

    lat = {c: [] for c in CONFIGS}
    t_round = []
    for r in range(n_rounds):
        t0 = time.time()
        for name, cfg in CONFIGS.items():
            apply(cfg)
            for im in imgs[warmup:]:
                t1 = time.perf_counter()
                m.predict(im, imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)
                lat[name].append((time.perf_counter() - t1) * 1000.0)
        t_round.append(time.time() - t0)
        log(f'第{r + 1}轮完成 ({t_round[-1]:.0f}s): '
            + '  '.join(f'{n} {np.median(lat[n][-n_frames:]):.2f}' for n in CONFIGS))

    head.set_temporal_enabled(False)
    head.set_query_overrides(encoder_queries=300, temporal_queries=0, output_topk=300)

    out = {}
    for name in CONFIGS:
        a = np.asarray(lat[name])
        out[name] = dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                         mean=float(a.mean()), n=len(a))
    out['_compare'] = {
        'b0_extra_ms_vs_a0': out['B0']['median'] - out['A0']['median'],
        'b0_saved_pct_vs_u0': 100 * (1 - out['B0']['median'] / out['U0']['median']),
        'a1_extra_ms_vs_a0': out['A1']['median'] - out['A0']['median'],
    }
    log(json.dumps(out, ensure_ascii=False, indent=1))
    with open(r'D:\lunwen\自研\query_prop_b0\logs\bench_b0_interleaved.json', 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
