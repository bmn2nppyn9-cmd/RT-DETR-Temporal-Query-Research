# -*- coding: utf-8 -*-
"""analyze_qd_o.py —— QD-O 阶段: 六层逐 Query 收敛深度 Oracle 分析
=====================================================================
对 record_qd_o.py 录制的六层逐层导出做:

  1. 三类"最早充分深度"分布:
     A. 最终层一致性深度 d_consistency (不用 GT, tau_box x eps_score 扫描)
     B. GT 正确深度 d_gt50 / d_gt75 (Oracle 上限, 逐类/尺寸/置信度分解)
     C. 背景/低价值 Query 稳定深度
  2. 逐层转移表 (L1/L3 正确 -> L6 仍正确 / L3 错误 -> 深层修正 / L3 正确 -> 深层变错)
  3. 混合深度 Oracle: O-consistency / O-GT50 / O-GT75 / Random-control / Confidence-control
     (+ 独立场景 calibration split 的逐层温度校准变体), 报告 mixed AP、E[d]、budget 节省
  4. 回答: 三层->六层增益由多少 hard Query 贡献

GT 只用于离线 Oracle 统计与上限分析, 不用于任何可部署策略。
输出: qd_oracle.json + qd_oracle_report.md

用法:
    python analyze_qd_o.py --preds preds_qd_o_u0.npz --suffix smoke
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

sys.path.insert(0, r'D:\lunwen\自研')
sys.path.insert(0, r'D:\lunwen\自研\query_prop_b0')
from eval_b0 import evaluate_ap, match_frame_tp  # noqa: E402
from eval_sap import build_timeline, load_gt  # noqa: E402

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
WORK = r'D:\lunwen\自研\query_depth_probe'
CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
               'traffic_light', 'stop_sign']

TAU_BOXES = [0.5, 0.7, 0.9]
EPS_SCORES = [0.02, 0.05, 0.10]
BG_TAUS = [0.01, 0.03, 0.05]


def log(m):
    print(m, flush=True)


def cxcywh_to_xyxy(cxcywh, w, h):
    """归一化 cxcywh (N,4) -> 像素 xyxy"""
    x = cxcywh[:, 0] * w
    y = cxcywh[:, 1] * h
    bw = cxcywh[:, 2] * w
    bh = cxcywh[:, 3] * h
    return np.stack([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2], -1).astype(np.float32)


def iou_pair(a, b):
    """a:(N,4) b:(N,4) xyxy -> (N,) 逐对 IoU"""
    lt = np.maximum(a[:, :2], b[:, :2])
    rb = np.minimum(a[:, 2:], b[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, 0] * wh[:, 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa + bb - inter, 1e-9)


def first_sustained_layer(ok):
    """ok:(6,N) bool, 每层是否满足 -> d(N,) 最早满足且持续到第 6 层的层 (0..5), 永不满足为 6"""
    suf = np.logical_and.accumulate(ok[::-1], axis=0)[::-1]
    d = np.where(suf.any(0), suf.argmax(0), 6).astype(np.int8)
    return d


def analyze(preds_path, ann, stream_index, suffix):
    log(f'加载 {preds_path}')
    z = np.load(preds_path, allow_pickle=False)
    l_box = z['l_box']      # (Nf,6,300,4) cxcywh sigmoid
    l_logit = z['l_logit']  # (Nf,6,300,8)
    l_ref = z['l_ref']      # (Nf,6,300,4)
    det_qidx = z['det_qidx']  # (Nf,300)
    det_cls = z['det_cls']    # (Nf,300)
    scene_a = z['scene']
    fidx_a = z['frame_idx']
    nf, nL, nq, _ = l_box.shape

    frames, scene_t, scene_files = build_timeline(stream_index)
    frame_key = {(f[0], f[1]): i for i, f in enumerate(frames)}
    gt = load_gt(ann)
    gidx = [frame_key[(int(scene_a[i]), int(fidx_a[i]))] for i in range(nf)]
    frame_files = [frames[g][3] for g in gidx]
    log(f'{nf} 帧 / {nL} 层 / {nq} query; GT {sum(len(v[0]) for v in gt.values())} 框')

    # 图像尺寸 (归一化框 -> 像素)
    with Image.open(os.path.join(DATA, 'images', frame_files[0])) as im:
        img_w, img_h = im.size
    log(f'图像尺寸 {img_w}x{img_h}')

    # 校验: L6 导出与最终 top-300 输出一致 (按 det_qidx/det_cls 逐位核对分数并验证降序)
    all_scores = 1.0 / (1.0 + np.exp(-l_logit))       # (Nf,6,300,8)
    l6_scores = all_scores[:, 5]                      # (Nf,300,8)
    flat = l6_scores.reshape(nf, -1)
    topk_flat = np.take_along_axis(flat, det_qidx.astype(np.int64) * 8 + det_cls.astype(np.int64), 1)
    ok_sorted = bool(np.all(np.diff(topk_flat, axis=1) <= 1e-7))
    log(f'L6 导出重建 top-300 分数降序: {ok_sorted} (导出与最终输出同源校验)')

    # ---------- 逐帧特征
    box6_xyxy = np.empty((nf, nq, 4), np.float32)
    c6 = np.argmax(l_logit[:, 5], -1)                       # (Nf,300)
    s6 = np.max(l6_scores, -1)                              # (Nf,300)
    for f in range(nf):
        box6_xyxy[f] = cxcywh_to_xyxy(l_box[f, 5], img_w, img_h)

    # ---------- A. d_consistency (tau_box x eps_score)
    d_cons = {}  # (tau,eps) -> (Nf,300) int8
    for tau in TAU_BOXES:
        for eps in EPS_SCORES:
            key = (tau, eps)
            acc = np.zeros((nf, nq), np.int8)
            for f in range(nf):
                b6 = box6_xyxy[f]
                ok = np.zeros((6, nq), bool)
                for li in range(6):
                    b_li = cxcywh_to_xyxy(l_box[f, li], img_w, img_h)
                    cls_li = np.argmax(l_logit[f, li], -1)
                    s_li = all_scores[f, li]  # (300,8) 该层各 query 各类别分数
                    s_li_c = s_li[np.arange(nq), c6[f]]
                    ok[li] = (cls_li == c6[f]) & (iou_pair(b_li, b6) >= tau) \
                        & (np.abs(s_li_c - s6[f]) <= eps)
                acc[f] = first_sustained_layer(ok)
            d_cons[key] = acc
    log(f'd_consistency 完成 ({len(d_cons)} 组阈值)')

    # ---------- B. d_gt50 / d_gt75
    d_gt50 = np.zeros((nf, nq), np.int8)
    d_gt75 = np.zeros((nf, nq), np.int8)
    tp_flag = np.zeros((nf, nq), bool)      # L6 最终 TP query
    gt_of = np.full((nf, nq), -1, np.int32)  # TP query 对应 GT 索引 (帧内)
    gt_cls_of = np.zeros((nf, nq), np.int8)
    gt_area_of = np.zeros((nf, nq), np.float32)
    for f in range(nf):
        gb, gc = gt[frame_files[f]]
        if len(gb) == 0:
            continue
        o = np.argsort(-s6[f], kind='stable')
        tpf, det2gt = match_frame_tp(box6_xyxy[f][o], s6[f][o], c6[f][o], gb, gc, 0.5)
        ok50 = np.zeros((6, nq), bool)
        ok75 = np.zeros((6, nq), bool)
        for li in range(6):
            b_li = cxcywh_to_xyxy(l_box[f, li], img_w, img_h)
            cls_li = np.argmax(l_logit[f, li], -1)
            g50 = np.zeros(nq, bool)
            g75 = np.zeros(nq, bool)
            for d_ in np.nonzero(tpf)[0]:
                q = o[d_]
                g = det2gt[d_]
                correct = cls_li[q] == gc[g]
                iou_g = iou_pair(b_li[q:q + 1], gb[g:g + 1])[0]
                g50[q] = g50[q] or (correct and iou_g >= 0.5)
                g75[q] = g75[q] or (correct and iou_g >= 0.75)
                if li == 5:
                    gt_of[f, q] = g
                    gt_cls_of[f, q] = gc[g]
                    gt_area_of[f, q] = (gb[g, 2] - gb[g, 0]) * (gb[g, 3] - gb[g, 1])
            ok50[li] = g50
            ok75[li] = g75
        d_gt50[f] = first_sustained_layer(ok50)
        d_gt75[f] = first_sustained_layer(ok75)
        tp_flag[f] = gt_of[f] >= 0
    n_tp = int(tp_flag.sum())
    log(f'd_gt 完成: 最终 TP query {n_tp}')

    # ---------- C. 背景/低价值 Query 稳定深度
    d_bg = {}  # tau -> (Nf,300) int8: 最大类分数从该层起持续 < tau
    m_max = all_scores.max(-1)                 # (Nf,6,300)
    for tau in BG_TAUS:
        acc = np.zeros((nf, nq), np.int8)
        for f in range(nf):
            ok = m_max[f] < tau
            acc[f] = first_sustained_layer(ok)
        d_bg[tau] = acc

    # ---------- 分布统计
    def hist(d):
        d = d.ravel()
        return {f'L{i+1}': int((d == i).sum()) for i in range(6)} | \
               {'never': int((d == 6).sum())}

    res = {}
    res['d_consistency'] = {f'tau{t:.1f}_eps{e:.2f}': hist(d_cons[(t, e)])
                            for t in TAU_BOXES for e in EPS_SCORES}
    res['d_consistency_t07_e05_split'] = {
        'tp': hist(d_cons[(0.7, 0.05)][tp_flag]),
        'non_tp': hist(d_cons[(0.7, 0.05)][~tp_flag]),
    }
    res['d_gt50'] = hist(d_gt50[tp_flag])
    res['d_gt75'] = hist(d_gt75[tp_flag])
    res['background_stable'] = {f'tau{t:.2f}': hist(d_bg[t]) for t in BG_TAUS}
    # 低价值子集 (L6 分数 < 0.05)
    lowq = s6 < 0.05
    res['background_stable_lowscore_only'] = {f'tau{t:.2f}': hist(d_bg[t][lowq]) for t in BG_TAUS}

    # d_gt50 分解: 逐类 / 尺寸 / 置信度
    gt_q = tp_flag
    per_cls = {}
    for c in range(8):
        m = gt_q & (gt_cls_of == c)
        per_cls[CLASS_NAMES[c]] = hist(d_gt50[m]) if m.any() else {'n': 0}
    sizes = {'small': gt_area_of < 32 ** 2, 'medium': (gt_area_of >= 32 ** 2) & (gt_area_of < 96 ** 2),
             'large': gt_area_of >= 96 ** 2}
    per_size = {k: hist(d_gt50[gt_q & m]) if (gt_q & m).any() else {'n': 0}
                for k, m in sizes.items()}
    med = float(np.median(s6[gt_q]))
    per_conf = {
        'high': hist(d_gt50[gt_q & (s6 >= med)]),
        'low': hist(d_gt50[gt_q & (s6 < med)]),
    }
    res['d_gt50_per_class'] = per_cls
    res['d_gt50_per_size'] = per_size
    res['d_gt50_per_conf'] = per_conf

    # ---------- 逐层转移表 (GT 口径, TP query)
    def correct_at(f, li):
        b_li = cxcywh_to_xyxy(l_box[f, li], img_w, img_h)
        cls_li = np.argmax(l_logit[f, li], -1)
        corr = np.zeros(nq, bool)
        qs = np.nonzero(tp_flag[f])[0]
        for q in qs:
            g = gt_of[f, q]
            corr[q] = (cls_li[q] == gt_cls_of[f, q]) \
                and iou_pair(b_li[q:q + 1], gt[frame_files[f]][0][g:g + 1])[0] >= 0.5
        return corr

    c1 = np.zeros(nf, np.int64)
    c2 = np.zeros(nf, np.int64)
    c3 = np.zeros(nf, np.int64)
    c3_err_fixed = np.zeros(nf, np.int64)   # L3 错误 -> L4/L5/L6 修正
    c3_ok_broke = np.zeros(nf, np.int64)    # L3 正确 -> L4/L5/L6 反而变错
    for f in range(nf):
        if not tp_flag[f].any():
            continue
        ok1 = correct_at(f, 0)
        ok2 = correct_at(f, 1)
        ok3 = correct_at(f, 2)
        ok4 = correct_at(f, 3)
        ok5 = correct_at(f, 4)
        tpq = tp_flag[f]
        c1[f] = (ok1 & tpq).sum()
        c2[f] = (ok2 & tpq).sum()
        c3[f] = (ok3 & tpq).sum()
        c3_err_fixed[f] = ((~ok3) & tpq & (ok4 | ok5)).sum()
        c3_ok_broke[f] = (ok3 & tpq & (~ok4 | ~ok5)).sum()
    res['transitions'] = {
        'n_tp': n_tp,
        'p_l1_correct_given_tp': float(c1.sum() / n_tp),
        'p_l2_correct_given_tp': float(c2.sum() / n_tp),
        'p_l3_correct_given_tp': float(c3.sum() / n_tp),
        'p_l3_err_fixed_by_deep': float(c3_err_fixed.sum() / n_tp),
        'p_l3_ok_broken_by_deep': float(c3_ok_broke.sum() / n_tp),
    }

    # ---------- 每层分数尺度诊断
    res['per_layer_mean_max_score'] = [float(m_max[:, li].mean()) for li in range(6)]

    # ---------- 混合深度 Oracle
    # 策略 -> 深度矩阵 (Nf,300) int8 (层 0..5)
    # 混合输出构造 = 真实系统结构: 每 query 用所选层的 box + 全类 logits, 展平 query×class 后
    # 取 top-300 (多类别重复行自然出现)。d=5 全部时与录制 U0 完全一致。
    default_key = (0.7, 0.05)
    strat = {
        'O_consistency_t07_e05': d_cons[default_key].copy(),
        'O_consistency_t05_e10': d_cons[(0.5, 0.10)].copy(),
        'O_GT50': np.where(tp_flag, d_gt50, 5).astype(np.int8),
        'O_GT75': np.where(tp_flag & (d_gt75 < 6), d_gt75, 5).astype(np.int8),
    }
    # 控制组使用与可部署策略 (O_consistency_t07_e05) 相同的每帧深度预算:
    # Random-control 随机分配; Confidence-control 按 L2 早期置信度分配 (高置信拿最浅层)
    base_depth = strat['O_consistency_t07_e05']
    rng = np.random.default_rng(0)
    rc = base_depth.copy()
    for f in range(nf):
        rng.shuffle(rc[f])
    strat['Random_control'] = rc
    cc = np.full((nf, nq), 5, np.int8)
    l2_sc = m_max[:, 1]  # L2 早期置信度
    for f in range(nf):
        budget = np.sort(base_depth[f])               # 升序: 最浅先
        order = np.argsort(-l2_sc[f], kind='stable')  # 高置信先
        cc[f, order] = budget
    strat['Confidence_control'] = cc

    def build_rec(depth, mask=None, calibrate=False):
        """按每 query 所选层构造混合输出 (真实系统结构: query×class 展平 top-300; conf 过滤)。

        calibrate=True 时分数用逐层仿射校准 (只应在独立 split 上评测)。
        返回 (rec, 与录制 det_qidx/det_cls 一致的帧数) 用于校验。
        """
        boxes, scores, clss, ptr = [], [], [], [0]
        idxs = np.arange(nf) if mask is None else np.nonzero(mask)[0]
        n_rows_same = 0
        for f in idxs:
            d = depth[f]
            b_sel = l_box[f, d, np.arange(nq)]              # (300,4) cxcywh
            lg_sel = l_logit[f, d, np.arange(nq)]           # (300,8)
            if calibrate:
                lg_flat = np.empty(nq * 8, np.float32)
                for li in range(6):
                    m = d == li
                    a, b = cal[li]
                    lg_flat.reshape(nq, 8)[m] = a * lg_sel[m] + b
                sc = 1.0 / (1.0 + np.exp(-lg_flat))
            else:
                sc = 1.0 / (1.0 + np.exp(-lg_sel)).reshape(-1)  # (2400,)
            topk_idx = np.argsort(-sc, kind='stable')[:300]
            qs = topk_idx // 8
            cs = topk_idx % 8
            if not calibrate:
                if np.array_equal(qs, det_qidx[f]) and np.array_equal(cs, det_cls[f]):
                    n_rows_same += 1
            b_xyxy = cxcywh_to_xyxy(b_sel[qs], img_w, img_h)
            s_sel = sc[topk_idx]
            keep = s_sel > 0.001
            boxes.append(b_xyxy[keep])
            scores.append(s_sel[keep])
            clss.append(cs[keep])
            ptr.append(ptr[-1] + int(keep.sum()))
        return {
            'box': np.concatenate(boxes).astype(np.float32),
            'score': np.concatenate(scores).astype(np.float32),
            'cls': np.concatenate(clss).astype(np.uint8),
            'ptr': np.array(ptr, np.int32),
            'scene': scene_a[idxs], 'frame_idx': fidx_a[idxs],
        }, n_rows_same

    log('评测 L6 (第六层全深度) 与混合策略 ...')
    l6_rec, n_same = build_rec(np.full((nf, nq), 5, np.int8))
    ap_l6 = evaluate_ap(l6_rec, gt, frame_files)
    res['ap_L6_from_export'] = ap_l6
    res['L6_topk_rows_identical_to_recorded'] = (n_same == nf)
    log(f'  L6_from_export: mAP50={ap_l6["map50"]:.5f} mAP={ap_l6["map"]:.5f} AP75={ap_l6["ap75"]:.5f} '
        f'(top-300 与录制一致 {n_same}/{nf} 帧)')

    for name, depth in strat.items():
        rec, _ = build_rec(depth)
        ap = evaluate_ap(rec, gt, frame_files)
        e_d = float(depth.mean() + 1.0)
        budget = float(depth.sum() + nf * nq)
        res[f'ap_{name}'] = ap
        res[f'budget_{name}'] = {
            'E_depth_layers': e_d,
            'query_layer_budget': budget,
            'savings_vs_6L_pct': float(100 * (1 - budget / (6 * nf * nq))),
        }
        log(f'  {name}: mAP50={ap["map50"]:.5f} mAP={ap["map"]:.5f} AP75={ap["ap75"]:.5f} '
            f'E[d]={e_d:.2f} 节省 {res[f"budget_{name}"]["savings_vs_6L_pct"]:.1f}%')

    # ---------- 校准变体 (scene parity split)
    even = (scene_a % 2) == 0
    odd = ~even
    cal = {}
    for li in range(6):
        x = l_logit[even, li].reshape(-1, 8)   # (Nq_cal, 8)
        y = l_logit[even, 5].reshape(-1, 8)
        xm = x.mean()
        ym = y.mean()
        a = float(((x - xm) * (y - ym)).sum() / max(((x - xm) ** 2).sum(), 1e-12))
        b = float(ym - a * xm)
        cal[li] = (a, b)
    res['calibration_affine'] = {f'L{li+1}': {'a': cal[li][0], 'b': cal[li][1]} for li in range(6)}

    # 仅对 odd 场景 (独立 split) 评测: 原始 vs 校准, 用 O_consistency_t07_e05
    odd_files = [frame_files[i] for i in np.nonzero(odd)[0]]
    rec_raw_odd, _ = build_rec(strat['O_consistency_t07_e05'], odd, calibrate=False)
    rec_cal_odd, _ = build_rec(strat['O_consistency_t07_e05'], odd, calibrate=True)
    res['ap_O_consistency_raw_on_odd_scenes'] = evaluate_ap(rec_raw_odd, gt, odd_files)
    res['ap_O_consistency_cal_on_odd_scenes'] = evaluate_ap(rec_cal_odd, gt, odd_files)
    log(f'  校准变体 (odd 场景): raw={res["ap_O_consistency_raw_on_odd_scenes"]["map50"]:.5f} '
        f'cal={res["ap_O_consistency_cal_on_odd_scenes"]["map50"]:.5f}')

    # ---------- 三层->六层增益的 hard Query 贡献
    deep = (d_gt50 >= 3) & tp_flag  # 需要 L4-L6 才首次正确的最终 TP
    res['hard_query_analysis'] = {
        'n_tp_need_l4_l6': int(deep.sum()),
        'share_of_tp_pct': float(100 * deep.sum() / max(n_tp, 1)),
        'n_tp_done_by_l3': int((d_gt50[tp_flag] <= 2).sum()),
    }

    # ---------- H1 支持条件汇总
    # 判定用可部署口径 (O_consistency_t07_e05, 不用 GT); GT oracle 作为正确性深度参照另报。
    t = res['transitions']
    odc = res['ap_O_consistency_t07_e05']
    bc = res['budget_O_consistency_t07_e05']
    od50 = res['ap_O_GT50']
    budget_gt50 = res['budget_O_GT50']
    res['H1_check'] = {
        '大量TP在L1-L3稳定正确': float(t['p_l3_correct_given_tp']),
        'hard_Query_由深层修正': float(t['p_l3_err_fixed_by_deep']),
        'mixed_vs_L6_gap_pp': float(100 * (ap_l6['map50'] - odc['map50'])),
        'budget_savings_pct': bc['savings_vs_6L_pct'],
        'random_control_map50': res['ap_Random_control']['map50'],
        'ref_gt50_mixed_vs_L6_gap_pp': float(100 * (ap_l6['map50'] - od50['map50'])),
        'ref_gt50_budget_savings_pct': budget_gt50['savings_vs_6L_pct'],
        'ref_gt50_tp_avg_depth_layers': float(d_gt50[tp_flag].mean() + 1.0),
    }

    out_json = os.path.join(WORK, f'qd_oracle{"_" + suffix if suffix else ""}.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log(f'已写入 {out_json}')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default=os.path.join(WORK, 'stream_preds', 'preds_qd_o_u0.npz'))
    ap.add_argument('--ann', default=os.path.join(DATA, 'annotations_val.json'))
    ap.add_argument('--stream-index', default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--suffix', default='')
    args = ap.parse_args()
    t0 = time.time()
    analyze(args.preds, args.ann, args.stream_index, args.suffix)
    log(f'总耗时 {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
