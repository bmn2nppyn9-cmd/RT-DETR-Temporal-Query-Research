# -*- coding: utf-8 -*-
"""eval_b0.py —— B0 探针四组对照评测
=====================================================
对 record_b0.py 录制的 npz 做:
  1. 同帧 AP 评测 (口径与 eval_sap.py --verify 一致: 同帧 GT 关联、全局分数排序、
     逐类 101 点插值), 额外输出 AP75 与逐类 AP
  2. 延迟统计 (median / P95, B0 相对 A0 的额外延迟, B0 相对 U0 的省时)
  3. temporal query 来源诊断 (仅 B0 有意义):
     - 每帧进入 top-300 的 temporal query 数 (均值/中位)
     - temporal query 置信度 (均值/中位)
     - IoU=0.5 类感知贪心匹配下 temporal TP 数
     - temporal 独立 TP (current 匹配不到、temporal 匹配到的 GT)
     - 逐类独立 TP
     - temporal 与 current 输出框重叠 (最大 IoU 的分布)
     - 场景首帧 vs 后续帧分开统计
  4. 复用 eval_sap.evaluate 的 sAP (L=0, 与已有协议一致)

GT 只用于评测统计, 不参与任何预测。

用法:
    python eval_b0.py --configs A0 A1 B0 U0 --suffix smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, r'D:\lunwen\自研')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from eval_sap import (IOU_THRS, build_timeline, evaluate as sap_evaluate,
                      greedy_tp, iou_matrix, load_gt, voc_ap)

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
               'traffic_light', 'stop_sign']


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------- AP (同帧口径)
def evaluate_ap(rec, gt, frame_files):
    """同帧 GT 关联的 AP; 返回 mAP50 / mAP50-95 / AP75 / 逐类 AP50 与 AP。

    与 eval_sap.py 的 verify 分支同口径: 每条预测关联自己帧的 GT, 全局按分数降序,
    (帧,类) 分组内贪心匹配, 逐类 101 点插值再平均。
    """
    box, score, cls, ptr = rec['box'], rec['score'], rec['cls'], rec['ptr']
    nf = len(rec['scene'])
    det_file = []
    for i in range(nf):
        det_file.extend([frame_files[i]] * (ptr[i + 1] - ptr[i]))

    order = np.argsort(-score, kind='stable')
    box_a, score_a, cls_a = box[order], score[order], cls[order]
    file_a = np.asarray(det_file, dtype=object)[order]

    fnames = sorted(set(file_a.tolist()))
    f2i = {f: i for i, f in enumerate(fnames)}
    fid = np.fromiter((f2i[f] for f in file_a.tolist()), np.int64, len(file_a))
    key = fid * 8 + cls_a
    gorder = np.argsort(key, kind='stable')
    sk = key[gorder]
    uniq, starts = np.unique(sk, return_index=True)
    ends = np.append(starts[1:], len(sk))

    n_gt = np.zeros(8, np.int64)
    for fn in set(frame_files):
        _, gc = gt[fn]
        if len(gc):
            n_gt += np.bincount(gc, minlength=8)

    cls_pos = [np.nonzero(cls_a == c)[0] for c in range(8)]
    aps = np.zeros((len(IOU_THRS), 8))
    for ti, thr in enumerate(IOU_THRS):
        tp = np.zeros(len(box_a), bool)
        for gi in range(len(uniq)):
            pos = gorder[starts[gi]:ends[gi]]
            u = uniq[gi]
            c = int(u % 8)
            f = fnames[int(u // 8)]
            gb, gc = gt[f]
            m = gc == c
            if not m.any():
                continue
            tp[pos] = greedy_tp(iou_matrix(box_a[pos], gb[m]), thr)
        for c in range(8):
            if n_gt[c] == 0:
                aps[ti, c] = np.nan
                continue
            idx = cls_pos[c]
            if len(idx) == 0:
                aps[ti, c] = 0.0
                continue
            csum = np.cumsum(tp[idx])
            rec_ = csum / n_gt[c]
            prec = csum / np.arange(1, len(idx) + 1)
            aps[ti, c] = voc_ap(rec_, prec)
    ap_per_cls = np.nanmean(aps, axis=1)          # 每个 IoU 档的 mAP
    i75 = int(np.argmin(np.abs(IOU_THRS - 0.75)))
    return {
        'map50': float(ap_per_cls[0]),
        'map': float(ap_per_cls.mean()),
        'ap75': float(ap_per_cls[i75]),
        'per_class_ap50': {CLASS_NAMES[c]: float(aps[0, c]) for c in range(8)},
        'per_class_ap': {CLASS_NAMES[c]: float(np.nanmean(aps[:, c])) for c in range(8)},
    }


# ---------------------------------------------------------------- temporal 诊断
def match_frame_tp(box, score, cls, gb, gc, thr=0.5):
    """单帧类感知贪心匹配; 返回 (tp flags, det->gt 索引)。box 需已按分数降序。"""
    tp = np.zeros(len(box), bool)
    det2gt = np.full(len(box), -1, np.int64)
    if len(box) == 0 or len(gb) == 0:
        return tp, det2gt
    taken = np.zeros(len(gb), bool)
    iou = iou_matrix(box, gb)
    iou = np.where(cls[:, None] == gc[None, :], iou, -1.0)
    for d in range(len(box)):
        row = np.where(taken, -1.0, iou[d])
        j = int(row.argmax())
        if row[j] >= thr:
            taken[j] = True
            tp[d] = True
            det2gt[d] = j
    return tp, det2gt


def diagnose_temporal(rec, gt, frame_files):
    box, score, cls, src, ptr = (rec['box'], rec['score'], rec['cls'],
                                 rec['src'], rec['ptr'])
    nf = len(rec['scene'])
    st = dict(n_temp=[], conf_temp=[], n_tp_temp=0, n_tp_cur=0, n_ind=0,
              overlap=[], first=dict(n_temp=[], n_ind=0),
              later=dict(n_temp=[], n_ind=0))
    per_cls_ind = np.zeros(8, np.int64)
    per_cls_tp_temp = np.zeros(8, np.int64)

    for i in range(nf):
        s, e = int(ptr[i]), int(ptr[i + 1])
        b, sc, cl, sr = box[s:e], score[s:e], cls[s:e], src[s:e]
        o = np.argsort(-sc, kind='stable')
        b, sc, cl, sr = b[o], sc[o], cl[o], sr[o]
        gb, gc = gt[frame_files[i]]
        is_first = int(rec['frame_idx'][i]) == 0

        tm = sr == 1
        cm = sr == 0
        nt = int(tm.sum())
        st['n_temp'].append(nt)
        (st['first'] if is_first else st['later'])['n_temp'].append(nt)
        if nt:
            st['conf_temp'].append(float(sc[tm].mean()))
            iou_cc = iou_matrix(b[tm], b[cm]) if cm.any() else np.zeros((nt, 0))
            st['overlap'].extend(iou_cc.max(axis=1).tolist() if cm.any() else [0.0] * nt)

        tp, det2gt = match_frame_tp(b, sc, cl, gb, gc)
        gt_by_temp = set(det2gt[tp & tm].tolist())
        gt_by_cur_all = set(det2gt[tp & cm].tolist())
        # current-only 重匹配: temporal 不在场时 current 能覆盖哪些 GT
        if cm.any():
            _, d2g_cur = match_frame_tp(b[cm], sc[cm], cl[cm], gb, gc)
            gt_cur_only = set(d2g_cur[d2g_cur >= 0].tolist())
        else:
            gt_cur_only = set()
        ind = gt_by_temp - gt_cur_only
        st['n_tp_temp'] += int((tp & tm).sum())
        st['n_tp_cur'] += int((tp & cm).sum())
        st['n_ind'] += len(ind)
        (st['first'] if is_first else st['later'])['n_ind'] += len(ind)
        for g in ind:
            per_cls_ind[gc[g]] += 1
        for d in np.nonzero(tp & tm)[0]:
            per_cls_tp_temp[cl[d]] += 1

    n_temp = np.asarray(st['n_temp'])
    ov = np.asarray(st['overlap']) if st['overlap'] else np.zeros(0)
    return {
        'temporal_per_frame_mean': float(n_temp.mean()),
        'temporal_per_frame_median': float(np.median(n_temp)),
        'temporal_conf_mean': float(np.mean(st['conf_temp'])) if st['conf_temp'] else 0.0,
        'temporal_conf_median': float(np.median(st['conf_temp'])) if st['conf_temp'] else 0.0,
        'tp_temporal_total': st['n_tp_temp'],
        'tp_current_total': st['n_tp_cur'],
        'independent_tp_total': st['n_ind'],
        'independent_tp_share_of_all_tp': float(st['n_ind'] / max(st['n_tp_temp'] + st['n_tp_cur'], 1)),
        'per_class_independent_tp': {CLASS_NAMES[c]: int(per_cls_ind[c]) for c in range(8)},
        'per_class_tp_temporal': {CLASS_NAMES[c]: int(per_cls_tp_temp[c]) for c in range(8)},
        'overlap_iou_mean': float(ov.mean()) if len(ov) else 0.0,
        'overlap_frac_iou_ge_0.5': float((ov >= 0.5).mean()) if len(ov) else 0.0,
        'first_frame': {
            'n_frames': len(st['first']['n_temp']),
            'temporal_per_frame_mean': float(np.mean(st['first']['n_temp'])) if st['first']['n_temp'] else 0.0,
            'independent_tp': st['first']['n_ind'],
        },
        'later_frames': {
            'n_frames': len(st['later']['n_temp']),
            'temporal_per_frame_mean': float(np.mean(st['later']['n_temp'])) if st['later']['n_temp'] else 0.0,
            'independent_tp': st['later']['n_ind'],
        },
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds-dir', default=r'D:\lunwen\自研\query_prop_b0\stream_preds')
    ap.add_argument('--configs', nargs='+', default=['a0', 'a1', 'b0', 'u0'])
    ap.add_argument('--ann', default=os.path.join(DATA, 'annotations_val.json'))
    ap.add_argument('--stream-index', default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--suffix', default='', help='输出文件名后缀, 如 smoke')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    frames, scene_t, scene_files = build_timeline(args.stream_index)
    frame_key = {(f[0], f[1]): i for i, f in enumerate(frames)}
    gt = load_gt(args.ann)
    log(f'时序 {len(frames)} 帧 / {len(scene_t)} 场景; GT {sum(len(v[0]) for v in gt.values())} 框')

    results = {}
    for cfg in args.configs:
        p = os.path.join(args.preds_dir, f'preds_{cfg}.npz')
        if not os.path.exists(p):
            log(f'[跳过] 缺 {p}')
            continue
        z = np.load(p, allow_pickle=False)
        rec = {k: z[k] for k in z.files}
        nf = len(rec['scene'])
        gidx = [frame_key[(int(rec['scene'][i]), int(rec['frame_idx'][i]))] for i in range(nf)]
        frame_files = [frames[g][3] for g in gidx]
        t0 = time.time()

        apres = evaluate_ap(rec, gt, frame_files)
        lat = rec['compute_ms'].astype(np.float64)
        lat_res = dict(median=float(np.median(lat)), p95=float(np.percentile(lat, 95)),
                       mean=float(lat.mean()))
        sap = sap_evaluate(rec, gt, frames, scene_t, scene_files, 0.0,
                           verify=False, frame_key=frame_key)
        entry = dict(ap=apres, latency=lat_res, sap_L0=sap, n_frames=nf)
        if 'src' in rec and (rec['src'] == 1).any():
            entry['temporal_diag'] = diagnose_temporal(rec, gt, frame_files)
        if 'nq_dec' in rec:
            nq = rec['nq_dec'].astype(np.int64)
            entry['nq_dec_counts'] = {int(v): int((nq == v).sum()) for v in np.unique(nq)}
        results[cfg] = entry
        log(f'{cfg}: mAP50={apres["map50"]:.5f} mAP={apres["map"]:.5f} AP75={apres["ap75"]:.5f} '
            f'延迟 {lat_res["median"]:.2f}/{lat_res["p95"]:.2f}ms ({time.time()-t0:.0f}s)')

    # 组间对比
    cmp_ = {}
    if 'a0' in results and 'b0' in results:
        cmp_['b0_minus_a0'] = {
            'map50_pp': 100 * (results['b0']['ap']['map50'] - results['a0']['ap']['map50']),
            'map_pp': 100 * (results['b0']['ap']['map'] - results['a0']['ap']['map']),
            'ap75_pp': 100 * (results['b0']['ap']['ap75'] - results['a0']['ap']['ap75']),
            'extra_latency_ms': results['b0']['latency']['median'] - results['a0']['latency']['median'],
        }
    if 'a1' in results and 'b0' in results:
        cmp_['b0_minus_a1'] = {
            'map50_pp': 100 * (results['b0']['ap']['map50'] - results['a1']['ap']['map50']),
            'map_pp': 100 * (results['b0']['ap']['map'] - results['a1']['ap']['map']),
        }
    if 'u0' in results and 'b0' in results:
        cmp_['b0_vs_u0'] = {
            'map50_pp': 100 * (results['b0']['ap']['map50'] - results['u0']['ap']['map50']),
            'time_saved_pct': 100 * (1 - results['b0']['latency']['median'] / results['u0']['latency']['median']),
        }
    results['_compare'] = cmp_

    out = args.out or os.path.join(
        r'D:\lunwen\自研\query_prop_b0',
        f'out_b0_eval{"_" + args.suffix if args.suffix else ""}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f'已写入 {out}')


if __name__ == '__main__':
    main()
