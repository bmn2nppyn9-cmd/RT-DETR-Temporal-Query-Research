# -*- coding: utf-8 -*-
"""
make_pair_subset.py —— B1a pair 训练子集构建
==============================================
从 stream_index_train.json(56 场景/39384 帧)构建连续 (t-1, t) 帧对:
  - 严格场景内, 禁跨场景传播;
  - 场景分层抽 20%(seed 20260923), 与基线 20% 图像子集同精神;
  - 输出: pairs_train.json(帧对)、t_frames_train.txt(t 帧清单, 对照组单图训练用
    同一 t 帧集合, 保证帧级一致)、all_frames_train.txt(pair 数据集 im_files,
    t 与 t-1 的并集)、val_sub20.txt(训练期验证 3012 帧快扫子集)、pair_subset_stats.json
    (Δt 统计, 混合 30/50/70 FPS 必须入报告)。

用法:
    python make_pair_subset.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqinfra.runtime import DATA_ROOT, PAIR_DIR
DATA = str(DATA_ROOT)
IMG_ROOT = os.path.join(DATA, 'argoverse-hd-yolo', 'images', 'train')
VAL_TXT = os.path.join(DATA, 'argoverse-hd-yolo', 'train_sub_val.txt')
HERE = str(PAIR_DIR)
PAIR_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20260923
FRAC = 0.2


def main():
    idx = json.load(open(os.path.join(DATA, 'argoverse-hd', 'stream_index_train.json'),
                         encoding='utf-8'))

    # 全部场景内连续对
    scene_pairs = []
    all_dts = []
    for s in idx['scenes']:
        files = [f['file'] for f in s['frames']]
        dts = [f.get('dt_ms') for f in s['frames']]
        pairs = [(files[i - 1], files[i]) for i in range(1, len(files))]
        scene_pairs.append(pairs)
        all_dts.extend(d for d in dts[1:] if d is not None)

    n_all = sum(len(p) for p in scene_pairs)
    # 场景分层 20%: 每场景等比例抽对(至少 1 对), 场景内均匀间隔抽取保证覆盖
    chosen = []
    for pairs in scene_pairs:
        if not pairs:
            continue
        k = max(1, int(round(len(pairs) * FRAC)))
        pos = np.linspace(0, len(pairs) - 1, k)
        pos = np.unique(np.round(pos).astype(int))
        take = pos  # 均匀间隔, 可复现
        chosen.extend(pairs[i] for i in take)

    # 重新按文件查 dt: 建 file -> dt_ms 映射
    file_dt = {}
    for s in idx['scenes']:
        for f in s['frames']:
            file_dt[f['file']] = f.get('dt_ms')

    dts_all = np.array([file_dt[c] for _, c in
                        [p for sp in scene_pairs for p in sp] if file_dt[c] is not None])
    dts_sub = np.array([file_dt[c] for _, c in chosen if file_dt[c] is not None])

    # 写出
    img = lambda fn: os.path.join(IMG_ROOT, fn).replace('/', os.sep)
    t_frames = sorted({img(c) for _, c in chosen})
    all_frames = sorted({img(p) for p, _ in chosen} | set(t_frames))

    # 缺标签检查: t 帧必须有 yolo 标签
    miss = [f for f in t_frames
            if not os.path.exists(f.replace(os.sep + 'images' + os.sep,
                                            os.sep + 'labels' + os.sep)[:-4] + '.txt')]
    if miss:
        raise FileNotFoundError(f'{len(miss)} 个 t 帧缺标签, 示例: {miss[:3]}')

    with open(os.path.join(HERE, 'pairs_train.json'), 'w', encoding='utf-8') as f:
        json.dump([[img(p), img(c)] for p, c in chosen], f, ensure_ascii=False)
    with open(os.path.join(HERE, 't_frames_train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(t_frames))
    with open(os.path.join(HERE, 'all_frames_train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_frames))

    # 训练期验证子集: val 全量清单的前 3012 帧(fraction=0.2 快扫代理, 已验证与全量
    # 最大差 <=0.02pp, 见项目记录)
    val_lines = open(VAL_TXT, encoding='utf-8').read().splitlines()
    with open(os.path.join(HERE, 'val_sub20.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines[:3012]))

    stats = {
        'seed': SEED, 'frac': FRAC,
        'scenes': len(scene_pairs), 'pairs_all': n_all, 'pairs_chosen': len(chosen),
        't_frames': len(t_frames), 'all_frames': len(all_frames),
        'dt_ms_all': {'min': float(dts_all.min()), 'median': float(np.median(dts_all)),
                      'mean': float(dts_all.mean()), 'max': float(dts_all.max()),
                      'p95': float(np.percentile(dts_all, 95))},
        'dt_ms_subset': {'min': float(dts_sub.min()), 'median': float(np.median(dts_sub)),
                         'mean': float(dts_sub.mean()), 'max': float(dts_sub.max()),
                         'p95': float(np.percentile(dts_sub, 95))},
        'note': '混合帧率 30/50/70 FPS; Δt 以毫秒计, 论文中关键帧间隔一律用时间差不用帧数',
    }
    with open(os.path.join(HERE, 'pair_subset_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"场景 {stats['scenes']}, 全量对 {n_all}, 抽样对 {len(chosen)}")
    print(f"t 帧 {len(t_frames)}, 并集帧 {len(all_frames)}")
    print(f"Δt 全量 ms: min {stats['dt_ms_all']['min']:.1f} / "
          f"中位 {stats['dt_ms_all']['median']:.1f} / max {stats['dt_ms_all']['max']:.1f}")
    print(f"Δt 子集 ms: min {stats['dt_ms_subset']['min']:.1f} / "
          f"中位 {stats['dt_ms_subset']['median']:.1f} / max {stats['dt_ms_subset']['max']:.1f}")
    print('已写出 pairs_train.json / t_frames_train.txt / all_frames_train.txt / '
          'val_sub20.txt / pair_subset_stats.json')


if __name__ == '__main__':
    main()
