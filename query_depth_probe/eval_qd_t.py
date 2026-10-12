# -*- coding: utf-8 -*-
"""eval_qd_t.py —— QD-T 阶段: Temporal 注入深度矩阵评测
=======================================================
对 record_qd_t.py 的每个 npz 报告:

  1. 同帧 AP: mAP50 / mAP50-95 / AP75 / 逐类 AP50 (口径与 eval_b0 一致)
  2. current-only AP / temporal-only AP (拆源)
  3. 相对 A0 的跨基线指标:
     - 真正新增 GT/TP: 本配置匹配到而 A0 全部输出匹配不到的 GT (全体 / 仅 temporal 来源)
     - A0 原有 GT 在混合输出中的丢失数
     - current Query 对 A0 的保持率 (A0 高分检测在 current 来源中的 IoU 0.5/0.75/0.95 存活率)
  4. temporal 诊断: 唯一 temporal Query 生存率 (unique qidx / M)、
     temporal/current 重叠率 (IoU>=0.5)、temporal 置信度
  5. 延迟: predict wall (compute_ms) 与 decoder-only CUDA Event (dec_ms) 的 median/P95

用法:
    python eval_qd_t.py --preds-dir stream_preds --configs a0 t31m25 t31m75 ... --suffix smoke
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
from eval_b0 import evaluate_ap, match_frame_tp  # noqa: E402
from eval_sap import build_timeline, iou_matrix, load_gt  # noqa: E402

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
WORK = r'D:\lunwen\自研\query_depth_probe'
CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
               'traffic_light', 'stop_sign']


def log(m):
    print(m, flush=True)


def rec_of(z):
    return {k: z[k] for k in z.files}


def sub_rec(rec, mask):
    """按输出条掩码切子集 rec (ptr 重建, 帧序保留; 无条的帧 ptr 连续)"""
    nf = len(rec['scene'])
    box, score, cls, ptr = [], [], [], [0]
    for i in range(nf):
        s, e = int(rec['ptr'][i]), int(rec['ptr'][i + 1])
        m = mask[s:e]
        box.append(rec['box'][s:e][m])
        score.append(rec['score'][s:e][m])
        cls.append(rec['cls'][s:e][m])
        ptr.append(ptr[-1] + int(m.sum()))
    return {
        'box': np.concatenate(box), 'score': np.concatenate(score), 'cls': np.concatenate(cls),
        'ptr': np.array(ptr, np.int32), 'scene': rec['scene'], 'frame_idx': rec['frame_idx'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds-dir', default=os.path.join(WORK, 'stream_preds'))
    ap.add_argument('--configs', nargs='+', default=[])
    ap.add_argument('--ann', default=os.path.join(DATA, 'annotations_val.json'))
    ap.add_argument('--stream-index', default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--suffix', default='')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    frames, scene_t, scene_files = build_timeline(args.stream_index)
    frame_key = {(f[0], f[1]): i for i, f in enumerate(frames)}
    gt = load_gt(args.ann)
    log(f'时序 {len(frames)} 帧; GT {sum(len(v[0]) for v in gt.values())} 框')

    # 参照: A0 (同目录同批录制)
    a0_path = os.path.join(args.preds_dir, 'preds_a0.npz')
    if not os.path.exists(a0_path):
        log('[错误] 缺 A0 参照 preds_a0.npz')
        sys.exit(1)
    a0 = rec_of(np.load(a0_path, allow_pickle=False))
    a0_gidx = [frame_key[(int(a0['scene'][i]), int(a0['frame_idx'][i]))] for i in range(len(a0['scene']))]
    a0_files = [frames[g][3] for g in a0_gidx]
    log(f'A0 参照: {len(a0["scene"])} 帧')

    # A0 逐帧 TP GT 集合 (只算一次)
    a0_tp_gt = [set() for _ in range(len(a0['scene']))]
    for i in range(len(a0['scene'])):
        s, e = int(a0['ptr'][i]), int(a0['ptr'][i + 1])
        b, sc, cl = a0['box'][s:e], a0['score'][s:e], a0['cls'][s:e]
        gb, gc = gt[a0_files[i]]
        if len(b) and len(gb):
            tpf, d2g = match_frame_tp(b, sc, cl, gb, gc, 0.5)
            a0_tp_gt[i] = set(d2g[tpf].tolist())

    if not args.configs:
        import glob
        tags = [os.path.splitext(os.path.basename(p))[0][6:] for p in
                sorted(glob.glob(os.path.join(args.preds_dir, 'preds_*.npz')))]
        args.configs = [t for t in tags if t != 'a0' and not t.startswith('qd_')]

    results = {}
    for cfg in ['a0'] + [c for c in args.configs if c != 'a0']:
        p = os.path.join(args.preds_dir, f'preds_{cfg}.npz')
        if not os.path.exists(p):
            log(f'[跳过] 缺 {p}')
            continue
        z = np.load(p, allow_pickle=False)
        rec = rec_of(z)
        nf = len(rec['scene'])
        gidx = [frame_key[(int(rec['scene'][i]), int(rec['frame_idx'][i]))] for i in range(nf)]
        frame_files = [frames[g][3] for g in gidx]
        assert frame_files == a0_files, 'A0 与配置帧集不一致'
        t0 = time.time()

        entry = {}
        apres = evaluate_ap(rec, gt, frame_files)
        entry['ap'] = apres
        lat = rec['compute_ms'].astype(np.float64)
        dec = rec['dec_ms'].astype(np.float64)
        entry['latency'] = dict(wall_median=float(np.median(lat)), wall_p95=float(np.percentile(lat, 95)))
        if (dec > 0).any():
            entry['decoder_cuda_ms'] = dict(median=float(np.median(dec[dec > 0])),
                                            p95=float(np.percentile(dec[dec > 0], 95)))

        # ---------- 逐帧统计
        agg = dict(n_new_all=0, n_new_tmp=0, n_a0_lost=0,
                   surv=[], overlap=[], conf_tmp=[], n_tmp=[], uniq_surv_den=0)
        keep_hit = {t: 0 for t in (0.5, 0.75, 0.95)}
        keep_total = 0
        n_tp_all = 0
        for i in range(nf):
            s, e = int(rec['ptr'][i]), int(rec['ptr'][i + 1])
            b, sc, cl = rec['box'][s:e], rec['score'][s:e], rec['cls'][s:e]
            sr = rec['src'][s:e]
            qx = rec['qidx'][s:e]
            gb, gc = gt[frame_files[i]]
            m = int(rec['nq_dec'][i]) - 300 if (rec['nq_dec'][i] > 300) else 0  # 注入的 temporal 数

            # TP 与跨 A0
            if len(b) and len(gb):
                tpf, d2g = match_frame_tp(b, sc, cl, gb, gc, 0.5)
                cfg_tp = set(d2g[tpf].tolist())
                n_tp_all += int(tpf.sum())
                agg['n_new_all'] += len(cfg_tp - a0_tp_gt[i])
                agg['n_a0_lost'] += len(a0_tp_gt[i] - cfg_tp)
                tm = sr == 1
                if tm.any():
                    tpf_t, d2g_t = match_frame_tp(b[tm], sc[tm], cl[tm], gb, gc, 0.5)
                    agg['n_new_tmp'] += len(set(d2g_t[tpf_t].tolist()) - a0_tp_gt[i])
            else:
                agg['n_a0_lost'] += len(a0_tp_gt[i])

            tm = sr == 1
            cm = sr == 0
            nt = int(tm.sum())
            agg['n_tmp'].append(nt)
            if nt:
                agg['conf_tmp'].append(float(sc[tm].mean()))
                iou_cc = iou_matrix(b[tm], b[cm]) if cm.any() else np.zeros((nt, 0))
                agg['overlap'].extend((iou_cc.max(1) if cm.any() else np.zeros(nt)).tolist())
            if m > 0:
                agg['uniq_surv_den'] += m
                agg['surv'].append(int(len(np.unique(qx[tm]))) if nt else 0)
            # current 对 A0 的保持率: A0 高分检测 (score>0.05) 在 current 来源中的同类 IoU 存活
            a0s, a0e = int(a0['ptr'][i]), int(a0['ptr'][i + 1])
            ab, asc, acl = a0['box'][a0s:a0e], a0['score'][a0s:a0e], a0['cls'][a0s:a0e]
            ah = asc > 0.05
            if ah.any() and cm.any():
                cb, csc, ccl = b[cm], sc[cm], cl[cm]
                taken = {t: np.zeros(len(cb), bool) for t in (0.5, 0.75, 0.95)}
                for row_idx in np.nonzero(ah)[0]:
                    cand = (ccl == acl[row_idx])
                    if not cand.any():
                        continue
                    iou = iou_matrix(cb[cand], ab[row_idx:row_idx + 1])[:, 0]
                    keep_total += 1
                    for t in (0.5, 0.75, 0.95):
                        cix = np.nonzero(cand)[0]
                        avail = iou >= t
                        avail &= ~taken[t][cand]
                        if avail.any():
                            j = cix[avail][int(iou[avail].argmax())]
                            taken[t][j] = True
                            keep_hit[t] += 1

        entry['cross_a0'] = {
            'new_gt_all': agg['n_new_all'],
            'new_gt_temporal_only': agg['n_new_tmp'],
            'a0_gt_lost': agg['n_a0_lost'],
            'tp_total': n_tp_all,
        }
        entry['current_keep_vs_a0'] = {f'iou{t}': keep_hit[t] / max(keep_total, 1) for t in (0.5, 0.75, 0.95)}
        entry['temporal_diag'] = {
            'per_frame_mean': float(np.mean(agg['n_tmp'])),
            'per_frame_median': float(np.median(agg['n_tmp'])),
            'conf_mean': float(np.mean(agg['conf_tmp'])) if agg['conf_tmp'] else 0.0,
            'overlap_iou_mean': float(np.mean(agg['overlap'])) if agg['overlap'] else 0.0,
            'overlap_frac_ge_0.5': float((np.asarray(agg['overlap']) >= 0.5).mean()) if agg['overlap'] else 0.0,
            'unique_query_survival_rate': float(np.sum(agg['surv']) / max(agg['uniq_surv_den'], 1))
            if agg['uniq_surv_den'] else 0.0,
            'unique_query_total': int(np.sum(agg['surv'])),
            'unique_query_denom': agg['uniq_surv_den'],
        }
        # ---------- 拆源 AP
        entry['ap_current_only'] = evaluate_ap(sub_rec(rec, rec['src'] == 0), gt, frame_files)
        entry['ap_temporal_only'] = evaluate_ap(sub_rec(rec, rec['src'] == 1), gt, frame_files)

        results[cfg] = entry
        log(f'{cfg}: mAP50={apres["map50"]:.5f} mAP={apres["map"]:.5f} AP75={apres["ap75"]:.5f} '
            f'cur={entry["ap_current_only"]["map50"]:.5f} tmp={entry["ap_temporal_only"]["map50"]:.5f} '
            f'new_gt={agg["n_new_all"]} (tmp源 {agg["n_new_tmp"]}) a0_lost={agg["n_a0_lost"]} '
            f'keep95={entry["current_keep_vs_a0"]["iou0.95"]:.3f} '
            f'surv={entry["temporal_diag"]["unique_query_survival_rate"]:.3f} '
            f'dec={entry.get("decoder_cuda_ms", {}).get("median", 0):.3f}ms ({time.time()-t0:.0f}s)')

    # ---------- 组间对比 (相对 A0)
    if 'a0' in results:
        a0ap = results['a0']['ap']['map50']
        for cfg, e in results.items():
            if cfg == 'a0':
                continue
            e['_vs_a0'] = {
                'map50_pp': 100 * (e['ap']['map50'] - a0ap),
                'map_pp': 100 * (e['ap']['map'] - results['a0']['ap']['map']),
                'ap75_pp': 100 * (e['ap']['ap75'] - results['a0']['ap']['ap75']),
            }

    out = args.out or os.path.join(WORK, f'temporal_depth{"_" + args.suffix if args.suffix else ""}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f'已写入 {out}')


if __name__ == '__main__':
    main()
