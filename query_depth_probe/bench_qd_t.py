# -*- coding: utf-8 -*-
"""bench_qd_t.py —— QD-T 阶段: 交错延迟基准 (随机轮转顺序, 消除热漂移)
====================================================================
每轮固定随机种子打乱配置顺序, 每配置跑 warm 帧 + 计时帧, 同时记录:
  - predict() wall time (perf_counter, 与 record 口径一致)
  - decoder-only CUDA Event 时间 (head._last_decoder_ms, 模型内计时)

temporal 配置每个配置块开头 reset state, 前 2 帧不计时 (建 state)。
输出: logs/bench_qd_t.json + 控制台表

用法:
    python bench_qd_t.py --configs A0 U0 T31m75 T23m75 T33m75 T3Om75 --rounds 2 --frames 100
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

sys.path.insert(0, r'C:\Users\TONY\ultralytics')
import ultralytics  # noqa: E402
assert ultralytics.__file__.lower().startswith(r'c:\users\tony\ultralytics\ultralytics'), \
    f'导入路径错误: {ultralytics.__file__}'

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
CKPT = r'D:\lunwen\自研\runs\detect\runs_rtdetr\r18_640_sub20_b4\weights\best.pt'
WORK = r'D:\lunwen\自研\query_depth_probe'

# 配置: name -> dict(eval_idx, temporal, src, inj, fuse, M)
BENCH_CONFIGS = {
    'A0': dict(eval_idx=2, temporal=False, src=3, inj=0, fuse=False, M=0),
    'U0': dict(eval_idx=5, temporal=False, src=3, inj=0, fuse=False, M=0),
    'T31m75': dict(eval_idx=2, temporal=True, src=3, inj=0, fuse=False, M=75),
    'T23m75': dict(eval_idx=2, temporal=True, src=2, inj=2, fuse=False, M=75),
    'T33m75': dict(eval_idx=2, temporal=True, src=3, inj=2, fuse=False, M=75),
    'T3Om75': dict(eval_idx=2, temporal=True, src=3, inj=2, fuse=True, M=75),
    'T31m25': dict(eval_idx=2, temporal=True, src=3, inj=0, fuse=False, M=25),
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
    ap.add_argument('--configs', nargs='+', default=list(BENCH_CONFIGS))
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--frames', type=int, default=150, help='每配置每轮计时帧数')
    ap.add_argument('--stream-index', default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--images', default=os.path.join(DATA, 'images'))
    ap.add_argument('--seed', type=int, default=20260923)
    args = ap.parse_args()

    with open(args.stream_index, encoding='utf-8') as f:
        idx = json.load(f)
    s0 = idx['scenes'][0]
    files = [os.path.join(args.images, fr['file']) for fr in s0['frames'][:args.frames + 4]]
    imgs = [load_bgr(p) for p in files]

    from ultralytics import RTDETR
    m = RTDETR(CKPT)
    head = find_head(m.model)
    assert head is not None
    head.set_temporal_enabled(False)
    head.time_decoder_cuda = False
    # warmup (temporal off)
    _ = m.predict(imgs[0], imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)[0]
    log('warmup 完成')

    acc = {c: dict(wall=[], dec=[]) for c in args.configs}
    for r in range(args.rounds):
        order = args.configs[:]
        rng = np.random.default_rng(args.seed + r)
        rng.shuffle(order)
        for cfg in order:
            c = BENCH_CONFIGS[cfg]
            head.set_temporal_enabled(False)
            head.time_decoder_cuda = False
            head.decoder.eval_idx = c['eval_idx']
            head.set_query_overrides(encoder_queries=300, temporal_queries=c['M'],
                                     output_topk=300, motion_alpha=0.0)
            head.set_temporal_config(source_depth=c['src'], injection_layer=c['inj'], fuse_after=c['fuse'])
            if c['temporal']:
                head.set_temporal_enabled(True)
            head.time_decoder_cuda = True  # 全部配置同口径计时 (含 A0/U0)
            head.reset_temporal_state()
            # 前 2 帧不计时: temporal 配置建 state, 非 temporal 配置缓冲热态
            _ = m.predict(imgs[0], imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)[0]
            _ = m.predict(imgs[1], imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)[0]
            for i in range(2, 2 + args.frames):
                t0 = time.perf_counter()
                _ = m.predict(imgs[i % len(imgs)], imgsz=640, conf=0.001, max_det=300,
                              device=0, verbose=False)[0]
                acc[cfg]['wall'].append((time.perf_counter() - t0) * 1000.0)
                acc[cfg]['dec'].append(head._last_decoder_ms)
            log(f'  round {r + 1}/{args.rounds} {cfg}: wall 中位 '
                f'{np.median(acc[cfg]["wall"][-args.frames:]):.2f}ms dec {np.median(acc[cfg]["dec"][-args.frames:]):.3f}ms')

    out = {}
    log('')
    log('=' * 74)
    log(f'{"config":10s} {"wall med":>8s} {"wall P95":>8s} {"wall mean":>9s} {"dec med":>7s} {"dec P95":>7s}')
    log('-' * 74)
    for cfg in args.configs:
        wall = np.asarray(acc[cfg]['wall'])
        dec = np.asarray(acc[cfg]['dec'])
        out[cfg] = dict(
            wall_median=float(np.median(wall)), wall_p95=float(np.percentile(wall, 95)),
            wall_mean=float(wall.mean()),
            wall_ci=float(1.96 * wall.std() / np.sqrt(len(wall))),
            decoder_cuda_median=float(np.median(dec)), decoder_cuda_p95=float(np.percentile(dec, 95)),
            decoder_cuda_mean=float(dec.mean()),
        )
        log(f'{cfg:10s} {out[cfg]["wall_median"]:8.2f} {out[cfg]["wall_p95"]:8.2f} '
            f'{out[cfg]["wall_mean"]:9.2f} {out[cfg]["decoder_cuda_median"]:7.3f} '
            f'{out[cfg]["decoder_cuda_p95"]:7.3f}')
    log('=' * 74)
    os.makedirs(os.path.join(WORK, 'logs'), exist_ok=True)
    op = os.path.join(WORK, 'logs', 'bench_qd_t.json')
    with open(op, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f'已写入 {op}')


if __name__ == '__main__':
    main()
