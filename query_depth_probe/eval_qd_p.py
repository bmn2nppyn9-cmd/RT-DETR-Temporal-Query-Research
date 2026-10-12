# -*- coding: utf-8 -*-
"""eval_qd_p.py —— QD-P 阶段: Query-depth schedule 评测
=====================================================
对 record_qd_p.py 的每个 npz 报告:

  1. 同帧 AP (mAP50 / mAP50-95 / AP75, 口径与 eval_b0 一致) + 相对 a0p 的 pp
  2. 实际执行统计: 平均 Query 深度 E[d] (1..3 层)、query-layer 预算与理论节省、
     各层实际宽度 vs 调度
  3. 延迟: predict wall / decoder-only CUDA / 整体 CUDA (median/P95)
  4. sAP (L=0)

用法:
    python eval_qd_p.py --suffix smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, r'D:\lunwen\自研')
sys.path.insert(0, r'D:\lunwen\自研\query_prop_b0')
from eval_b0 import evaluate_ap  # noqa: E402
from eval_sap import build_timeline, evaluate as sap_evaluate, load_gt  # noqa: E402

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
WORK = r'D:\lunwen\自研\query_depth_probe'


def log(m):
    print(m, flush=True)


def rec_of(z):
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds-dir', default=os.path.join(WORK, 'stream_preds'))
    ap.add_argument('--configs', nargs='+', default=[])
    ap.add_argument('--a0-tag', default='a0p')
    ap.add_argument('--ann', default=os.path.join(DATA, 'annotations_val.json'))
    ap.add_argument('--stream-index', default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--suffix', default='')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    frames, scene_t, scene_files = build_timeline(args.stream_index)
    frame_key = {(f[0], f[1]): i for i, f in enumerate(frames)}
    gt = load_gt(args.ann)
    log(f'时序 {len(frames)} 帧; GT {sum(len(v[0]) for v in gt.values())} 框')

    a0_path = os.path.join(args.preds_dir, f'preds_{args.a0_tag}.npz')
    if not os.path.exists(a0_path):
        log(f'[错误] 缺基线 preds_{args.a0_tag}.npz')
        sys.exit(1)
    a0 = rec_of(np.load(a0_path, allow_pickle=False))
    a0_gidx = [frame_key[(int(a0['scene'][i]), int(a0['frame_idx'][i]))] for i in range(len(a0['scene']))]
    a0_files = [frames[g][3] for g in a0_gidx]
    ap_a0 = evaluate_ap(a0, gt, a0_files)
    log(f'基线 {args.a0_tag}: mAP50={ap_a0["map50"]:.5f} mAP={ap_a0["map"]:.5f} AP75={ap_a0["ap75"]:.5f}')

    if not args.configs:
        import glob
        tags = [os.path.splitext(os.path.basename(p))[0][6:] for p in
                sorted(glob.glob(os.path.join(args.preds_dir, 'preds_p*.npz')))]
        args.configs = tags

    results = {}
    a0_lat = a0['compute_ms'].astype(np.float64)
    a0_dec = a0['dec_ms'].astype(np.float64)
    a0_mod = a0['model_ms'].astype(np.float64)
    sap_a0 = sap_evaluate(a0, gt, frames, scene_t, scene_files, 0.0, verify=False, frame_key=frame_key)
    results[args.a0_tag] = {
        'ap': ap_a0,
        'vs_a0_pp': {'map50': 0.0, 'map': 0.0, 'ap75': 0.0},
        'execution': {'E_depth_layers': 3.0, 'budget_query_layers': 3.0 * len(a0['scene']) * 300,
                      'savings_vs_3L_pct': 0.0, 'widths_actual': {300: int(len(a0['scene']))}},
        'budget_theoretical': {'query_layers': 900, 'savings_vs_3L_pct': 0.0},
        'latency': {
            'wall_median': float(np.median(a0_lat)),
            'wall_p95': float(np.percentile(a0_lat, 95)),
            'decoder_cuda_median': float(np.median(a0_dec)),
            'model_cuda_median': float(np.median(a0_mod)),
            'model_cuda_p95': float(np.percentile(a0_mod, 95)),
        },
        'sap_L0': {k: sap_a0[k] for k in sap_a0 if k in ('sap', 'sap50')},
    }
    for cfg in args.configs:
        p = os.path.join(args.preds_dir, f'preds_{cfg}.npz')
        if not os.path.exists(p):
            log(f'[跳过] 缺 {p}')
            continue
        z = np.load(p, allow_pickle=False)
        rec = rec_of(z)
        nf = len(rec['scene'])
        gidx = [frame_key[(int(rec['scene'][i]), int(rec['frame_idx'][i]))] for i in range(nf)]
        frame_files = [frames[g][3] for g in gidx]
        assert frame_files == a0_files, '基线与配置帧集不一致'
        t0 = time.time()

        entry = {}
        apres = evaluate_ap(rec, gt, frame_files)
        entry['ap'] = apres
        entry['vs_a0_pp'] = {
            'map50': float(100 * (apres['map50'] - ap_a0['map50'])),
            'map': float(100 * (apres['map'] - ap_a0['map'])),
            'ap75': float(100 * (apres['ap75'] - ap_a0['ap75'])),
        }
        # 实际执行统计
        ann = rec['exit_ann']                    # (Nf, 300) -1/0/1
        depth = np.where(ann >= 0, ann + 1, 3)   # 每 query 实际层数
        e_d = float(depth.mean())
        entry['execution'] = {
            'E_depth_layers': e_d,
            'budget_query_layers': float(depth.sum()),
            'savings_vs_3L_pct': float(100 * (1 - e_d / 3)),
            'widths_actual': {int(w): int((rec['widths'] == w).all(1).sum()) for w in
                              np.unique(rec['widths'].reshape(-1))},
        }
        # 理论节省 (由 schedule 标签推导)
        if cfg.startswith('p1_'):
            theo = 525
        elif cfg.startswith('p2_'):
            theo = 600
        elif cfg.startswith('p3_'):
            theo = 450
        else:
            theo = 900
        entry['budget_theoretical'] = {'query_layers': theo, 'savings_vs_3L_pct': float(100 * (1 - theo / 900))}
        # 延迟
        entry['latency'] = {
            'wall_median': float(np.median(rec['compute_ms'])),
            'wall_p95': float(np.percentile(rec['compute_ms'], 95)),
            'decoder_cuda_median': float(np.median(rec['dec_ms'])),
            'model_cuda_median': float(np.median(rec['model_ms'])),
            'model_cuda_p95': float(np.percentile(rec['model_ms'], 95)),
        }
        # sAP (L=0)
        sap = sap_evaluate(rec, gt, frames, scene_t, scene_files, 0.0, verify=False, frame_key=frame_key)
        entry['sap_L0'] = {k: sap[k] for k in sap if k in ('sap', 'sap50')}
        results[cfg] = entry
        log(f'{cfg}: mAP50={apres["map50"]:.5f} ({entry["vs_a0_pp"]["map50"]:+.2f}pp) '
            f'mAP={apres["map"]:.5f} AP75={apres["ap75"]:.5f} E[d]={e_d:.2f} '
            f'实际省 {entry["execution"]["savings_vs_3L_pct"]:.1f}% / 理论省 {entry["budget_theoretical"]["savings_vs_3L_pct"]:.1f}% '
            f'wall {entry["latency"]["wall_median"]:.2f}ms dec {entry["latency"]["decoder_cuda_median"]:.3f}ms '
            f'model {entry["latency"]["model_cuda_median"]:.2f}ms ({time.time()-t0:.0f}s)')

    out = args.out or os.path.join(WORK, f'query_depth_schedule{"_" + args.suffix if args.suffix else ""}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f'已写入 {out}')


if __name__ == '__main__':
    main()
