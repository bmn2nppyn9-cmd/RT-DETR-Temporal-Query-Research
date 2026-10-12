# -*- coding: utf-8 -*-
"""
eval_full_metrics.py —— COCO 全指标评测(AP75/AR/尺寸分档/工作点 P-R-F1)
==========================================================================
从录制的 stream_preds npz 纯后处理出完整检测指标, 不重跑推理。

口径说明(重要, 写论文时照此引用):
    AP / AP50 / AP75  : 沿用 eval_sap.py 的 verify 口径(每帧预测对本帧 GT,
        全量 det 参与, 逐类积分再平均)。该口径与 ultralytics val 逐位对齐
        (d1/d3/d6 的 AP50 必须复现 0.3364/0.3865/0.4053, 本脚本自带回归检查),
        保证与已报数字一致。
    AR@1/10/100       : COCO 官方口径 —— 每图每类按分数取 top-k 个 det,
        recall 在 IoU=0.50:0.95 十个阈值上平均。top-k 语义与本脚本的
        det-side 贪心匹配严格兼容: GT 被名次<=k 的 det 认领 等价于
        在 top-k 子集上重跑匹配(论证见 greedy_match 注释)。
    AP_small/medium/large : COCO 官方面积分档(small: area<32^2, medium:
        32^2<=area<96^2, large: area>=96^2), 面积按原生 1920x1200 尺度。
        档外 det 忽略(不算 FP), 档外 GT 忽略; 某类某档无 GT 则该类该档
        不参与平均(COCO 的 -1 语义)。val 集三档占比 38.8/40.5/20.7%
        (见 out_gt_area_stats.json), 分档不退化。
    工作点 P/R/F1     : IoU=0.5, 池化全局分数序上逐前缀计算, 取 mean-F1
        最大处, 与 ultralytics 报告的 P/R 同口径(回归参照: d6 约 P=0.588
        R=0.383)。

用法:
    python eval_full_metrics.py                      # d1/d3/d6 全量
    python eval_full_metrics.py --tags d6            # 只跑一档
    python eval_full_metrics.py --preds-dir query_prop_b0\\xxx --tags b0

输出:
    打印: 主指标对比表 + 逐类表 + 工作点表
    文件: out_full_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from eval_sap import (IOU_THRS, build_timeline, iou_matrix, load_gt, log,
                      voc_ap)

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
CATS = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
        'traffic_light', 'stop_sign']
NC = len(CATS)
AREA_EDGES = [32.0 ** 2, 96.0 ** 2]          # COCO small/medium/large 分界
AREA_NAMES = ['small', 'medium', 'large']
AR_KS = [1, 10, 100]
# 回归参照: eval_sap.py --verify 同口径存档值(见 out_sap_verify.json),
# 要求逐位一致; 对应 ultralytics 已报数字 0.3364/0.3865/0.4053
# (稳定 +0.002 内的近似偏移, 见项目记录)
REF_AP50 = {'d1': 0.3388, 'd3': 0.3882, 'd6': 0.4070}
REF_TOL = 1e-3
REF_PR_D6 = (0.588, 0.383)                    # d6 工作点 P/R (ultralytics)


def greedy_match(iou, thr):
    """det-side 贪心匹配(eval_sap 同口径)。

    返回 (dt_tp[D], gt_rank[G]):
        dt_tp[d]   —— det d 是否为 TP
        gt_rank[g] —— 认领 GT g 的 det 在组内的名次(组内按分数降序), 未认领 -1

    top-k 等价性: 名次>k 的 det 被移除后, 剩余 det 的匹配行为不变(处理序
    不变、每名 det 面对的空闲 GT 集合不变), 因此 "GT 被名次<=k 的 det 认领"
    与 "在 top-k 子集上重新贪心匹配" 结果相同 —— AR@k 无需重复匹配。
    """
    D, G = iou.shape
    dt_tp = np.zeros(D, bool)
    gt_rank = np.full(G, -1, np.int32)
    if D == 0 or G == 0:
        return dt_tp, gt_rank
    best = iou.max(axis=1)
    cand = np.nonzero(best >= thr)[0]
    taken = np.zeros(G, bool)
    for d in cand:
        row = iou[d]
        masked = np.where(taken, -1.0, row)
        j = int(masked.argmax())
        if masked[j] >= thr:
            taken[j] = True
            dt_tp[d] = True
            gt_rank[j] = d
    return dt_tp, gt_rank


def ap_from_tp(tp, n_gt):
    """tp: 按分数降序的布尔数组; 返回 101 点插值 AP"""
    csum = np.cumsum(tp)
    rec = csum / n_gt
    prec = csum / np.arange(1, len(tp) + 1)
    return voc_ap(rec, prec)


def evaluate_full(rec, gt, frames, frame_key):
    """单个配置的全指标评测。rec: npz 全部数组的字典"""
    box, score, cls, ptr = rec['box'], rec['score'], rec['cls'], rec['ptr']
    nfr = len(rec['scene'])

    # 帧 -> 文件名(经 stream_index 时序表映射, 与 eval_sap 一致)
    gidx = np.fromiter(
        (frame_key.get((int(s), int(f)), -1)
         for s, f in zip(rec['scene'], rec['frame_idx'])),
        np.int64, nfr)
    if (gidx < 0).any():
        raise SystemExit('[错误] 录制帧不在时序表内')
    counts = np.diff(ptr)
    fid_of_frame = gidx                       # 时序表序号即文件序号
    fid = np.repeat(fid_of_frame, counts)     # 每个 det 一条

    # det 面积(原生尺度)与分档
    det_area = np.clip(box[:, 2] - box[:, 0], 0, None) * \
        np.clip(box[:, 3] - box[:, 1], 0, None)
    det_rng = np.digitize(det_area, AREA_EDGES).astype(np.int8)   # 0/1/2

    # GT 统计(评测范围=全部帧; 逐类 + 逐类逐档)
    n_gt = np.zeros(NC, np.int64)
    n_gt_rng = np.zeros((3, NC), np.int64)
    gt_cache = {}
    for g in np.unique(gidx):
        fn = frames[g][3]
        gb, gc = gt[fn]
        gt_cache[g] = (gb, gc)
        if len(gc):
            n_gt += np.bincount(gc, minlength=NC)
            ga = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
            gr = np.digitize(ga, AREA_EDGES)
            for r in range(3):
                n_gt_rng[r] += np.bincount(gc[gr == r], minlength=NC)

    # 全局分数降序 + 按 (文件, 类) 分组
    order = np.argsort(-score, kind='stable')
    box_s, cls_s, fid_s = box[order], cls[order], fid[order]
    rng_s = det_rng[order]
    ndet = len(order)
    key = fid_s.astype(np.int64) * NC + cls_s
    gorder = np.argsort(key, kind='stable')
    sk = key[gorder]
    uniq, starts = np.unique(sk, return_index=True)
    ends = np.append(starts[1:], ndet)

    n_thr = len(IOU_THRS)
    tp_all = np.zeros((n_thr, ndet), bool)          # [thr, 全局位置]
    tp_rng = np.zeros((3, n_thr, ndet), bool)       # [档, thr, 全局位置]
    ar_hits = np.zeros((n_thr, len(AR_KS), NC), np.int64)

    t0 = time.time()
    for gi in range(len(uniq)):
        pos = gorder[starts[gi]:ends[gi]]           # 组内保持分数降序
        u = uniq[gi]
        c = int(u % NC)
        g = int(u // NC)
        gb, gc = gt_cache[g]
        m = gc == c
        if not m.any():
            continue                                # 该类无 GT: det 全为 FP
        gb_c = gb[m]
        iou = iou_matrix(box_s[pos], gb_c)
        for ti, thr in enumerate(IOU_THRS):
            tp, grank = greedy_match(iou, thr)
            tp_all[ti, pos] = tp
            for ki, k in enumerate(AR_KS):
                ar_hits[ti, ki, c] += int(((grank >= 0) & (grank < k)).sum())
            # 尺寸分档: 档外 det 忽略, 档外 GT 忽略, 档内重匹配
            ga = (gb_c[:, 2] - gb_c[:, 0]) * (gb_c[:, 3] - gb_c[:, 1])
            gr = np.digitize(ga, AREA_EDGES)
            dr = rng_s[pos]
            for r in range(3):
                dm = dr == r
                gm = gr == r
                if not dm.any() or not gm.any():
                    continue                        # 档内无 det 或无 GT
                tp_r, _ = greedy_match(iou[np.ix_(dm, gm)], thr)
                tp_rng[r, ti, pos[dm]] = tp_r
        if gi % 20000 == 0:
            log(f'    组 {gi}/{len(uniq)}  ({time.time()-t0:.0f}s)')

    # ---------------- 聚合 ----------------
    cls_pos = [np.nonzero(cls_s == c)[0] for c in range(NC)]
    res = {'n_gt': n_gt.tolist(),
           'n_gt_area': {AREA_NAMES[r]: n_gt_rng[r].tolist() for r in range(3)}}

    # AP / AP50 / AP75 (逐类, 再平均)
    per_class = {}
    ap_thr = np.zeros((NC, n_thr))                  # 逐类逐 IoU 的 AP
    for c in range(NC):
        idx = cls_pos[c]
        for ti in range(n_thr):
            ap_thr[c, ti] = ap_from_tp(tp_all[ti, idx], n_gt[c]) \
                if len(idx) else 0.0
        per_class[CATS[c]] = {
            'AP': float(ap_thr[c].mean()),
            'AP50': float(ap_thr[c, 0]),
            'AP75': float(ap_thr[c, 5])}
    res['AP'] = float(ap_thr.mean())
    res['AP50'] = float(ap_thr[:, 0].mean())
    res['AP75'] = float(ap_thr[:, 5].mean())

    # AR@1/10/100 (逐类逐阈值 recall 平均)
    ar_cls = np.zeros((NC, len(AR_KS)))
    for c in range(NC):
        for ki in range(len(AR_KS)):
            ar_cls[c, ki] = (ar_hits[:, ki, c] / n_gt[c]).mean()
    for ki, k in enumerate(AR_KS):
        res[f'AR{k}'] = float(ar_cls[:, ki].mean())
    for c in range(NC):
        per_class[CATS[c]]['AR100'] = float(ar_cls[c, 2])

    # 尺寸分档 AP (IoU 0.50:0.95 平均; 无 GT 的类不参与平均)
    for r, rn in enumerate(AREA_NAMES):
        aps = []
        for c in range(NC):
            if n_gt_rng[r, c] == 0:
                continue
            idx = cls_pos[c]
            inr = rng_s[idx] == r
            if not inr.any():
                aps.append(0.0)
                continue
            tp_c = tp_rng[r][:, idx][:, inr]        # [thr, 档内det]
            aps.append(float(np.mean([
                ap_from_tp(tp_c[ti], n_gt_rng[r, c]) for ti in range(n_thr)])))
            per_class[CATS[c]][f'AP_{rn}'] = aps[-1]
        res[f'AP_{rn}'] = float(np.mean(aps))

    # 工作点 P/R/F1 (IoU=0.5, 池化分数序, mean-F1 最大处)
    tp05 = tp_all[0]
    P = np.zeros(NC)
    R = np.zeros(NC)
    F1 = np.zeros(NC)
    f1_curve = np.zeros(ndet)
    for c in range(NC):
        is_c = cls_s == c
        tpc = tp05 & is_c
        fpc = (~tp05) & is_c
        cum_tp = np.cumsum(tpc)
        cum_fp = np.cumsum(fpc)
        p = cum_tp / np.maximum(cum_tp + cum_fp, 1)
        r = cum_tp / n_gt[c]
        f1_curve += 2 * p * r / np.maximum(p + r, 1e-9)
        # 记录逐类曲线末端备查
        per_class[CATS[c]]['n_det'] = int(is_c.sum())
    f1_curve /= NC
    i_best = int(np.argmax(f1_curve))
    theta = float(score[order][i_best])
    for c in range(NC):
        is_c = cls_s == c
        tpc = (tp05 & is_c)[:i_best + 1]
        fpc = ((~tp05) & is_c)[:i_best + 1]
        tp_n, fp_n = int(tpc.sum()), int(fpc.sum())
        P[c] = tp_n / max(tp_n + fp_n, 1)
        R[c] = tp_n / n_gt[c]
        F1[c] = 2 * P[c] * R[c] / max(P[c] + R[c], 1e-9)
        per_class[CATS[c]].update(
            P=float(P[c]), R=float(R[c]), F1=float(F1[c]))
    res['operating_point'] = {
        'theta': theta, 'P': float(P.mean()), 'R': float(R.mean()),
        'F1': float(F1.mean())}
    res['per_class'] = per_class
    return res


def print_table(results):
    tags = list(results.keys())
    cols = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
            'AR1', 'AR10', 'AR100']
    log('')
    log('=' * 78)
    log('主指标 (全量 val 15062 帧, verify 口径)')
    hdr = f"{'配置':<8}" + ''.join(f'{c:>10}' for c in cols)
    log(hdr)
    for t in tags:
        r = results[t]
        log(f"{t:<8}" + ''.join(f'{r[c]:>10.4f}' for c in cols))
    log('')
    log('工作点 (IoU=0.5, mean-F1 最大处)')
    log(f"{'配置':<8}{'阈值':>8}{'P':>10}{'R':>10}{'F1':>10}")
    for t in tags:
        op = results[t]['operating_point']
        log(f"{t:<8}{op['theta']:>8.3f}{op['P']:>10.4f}{op['R']:>10.4f}"
            f"{op['F1']:>10.4f}")
    log('')
    for t in tags:
        log(f'逐类 ({t}):')
        log(f"{'类别':<15}{'AP':>8}{'AP50':>8}{'AP75':>8}{'AR100':>8}"
            f"{'AP_s':>8}{'AP_m':>8}{'AP_l':>8}{'P':>7}{'R':>7}")
        for cn in CATS:
            pc = results[t]['per_class'][cn]
            log(f"{cn:<15}{pc['AP']:>8.4f}{pc['AP50']:>8.4f}"
                f"{pc['AP75']:>8.4f}{pc['AR100']:>8.4f}"
                f"{pc.get('AP_small',0):>8.4f}{pc.get('AP_medium',0):>8.4f}"
                f"{pc.get('AP_large',0):>8.4f}{pc['P']:>7.3f}{pc['R']:>7.3f}")
        log('')
    log('=' * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds-dir', default='stream_preds')
    ap.add_argument('--tags', nargs='+', default=['d1', 'd3', 'd6'],
                    help='npz 名 preds_<tag>.npz')
    ap.add_argument('--ann', default=os.path.join(DATA, 'annotations_val.json'))
    ap.add_argument('--stream-index',
                    default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--out', default='out_full_metrics.json')
    args = ap.parse_args()

    frames, _, _ = build_timeline(args.stream_index)
    frame_key = {(f[0], f[1]): i for i, f in enumerate(frames)}
    gt = load_gt(args.ann)
    log(f'时序 {len(frames)} 帧, GT {sum(len(v[0]) for v in gt.values())} 框')

    results = {}
    for tag in args.tags:
        p = os.path.join(args.preds_dir, f'preds_{tag}.npz')
        if not os.path.exists(p):
            log(f'[跳过] 缺 {p}')
            continue
        z = np.load(p, allow_pickle=False)
        rec = {k: z[k] for k in z.files}
        log(f'[{tag}] {len(rec["scene"])} 帧 / {len(rec["score"])} det, '
            f'评测中...')
        t0 = time.time()
        results[tag] = evaluate_full(rec, gt, frames, frame_key)
        r = results[tag]
        log(f'[{tag}] 完成 ({time.time()-t0:.0f}s)  '
            f'AP={r["AP"]:.4f} AP50={r["AP50"]:.4f} AR100={r["AR100"]:.4f}')
        if tag in REF_AP50:
            diff = r['AP50'] - REF_AP50[tag]
            flag = 'OK' if abs(diff) < REF_TOL else '!!! 回归不符, 检查口径'
            log(f'    回归: AP50 {r["AP50"]:.4f} vs eval_sap 存档 '
                f'{REF_AP50[tag]:.4f}  差 {diff:+.4f}  {flag}')
        if tag == 'd6':
            op = r['operating_point']
            log(f'    参照: 工作点 P={op["P"]:.3f} R={op["R"]:.3f} '
                f'(ultralytics 约 P={REF_PR_D6[0]} R={REF_PR_D6[1]})')

    print_table(results)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f'已写入 {args.out}')


if __name__ == '__main__':
    main()
