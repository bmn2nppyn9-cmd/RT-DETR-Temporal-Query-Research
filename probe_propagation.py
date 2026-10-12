# -*- coding: utf-8 -*-
"""
probe_propagation.py —— 路径B前置诊断: 前帧检测对当前帧的预测力
================================================================
回答的问题:
    t-1 帧的检测框, 能多大程度预测 t 帧的 GT?
    若预测力高 -> 把前帧 query 作为当前帧 decoder 的先验有搞头
    若预测力低 -> 传播必须依赖内容嵌入/记忆库, 仅靠框的位置不够

三个探针, 全部在同一批帧上评(帧序号>=2, 保证 t-1 与 t-2 都存在):
    model   : 模型自己在 t 帧的预测        (看过 t 帧, 参照上界)
    copy    : 直接把 t-1 的框当作 t 的预测  (零阶保持, alpha=0)
    extrap  : 用 t-2->t-1 的运动矢量外推    (alpha 阻尼, 防过冲)

评估口径:
    与 eval_sap.py 中已验证的 AP 实现同构 —— 逐类积分再平均、跳过无 GT 的类、
    101 点插值。GT 分母严格等于参与评测的帧集, 三种探针完全一致, 因此
    "model vs copy vs extrap" 的对比是公平的。

额外指标 recall@IoU0.5:
    不依赖分数排序, 更干净地反映"位置信息量"。

用法:
    python probe_propagation.py                      # 用 depth=6 的录制
    python probe_propagation.py --depth 3
    python probe_propagation.py --alphas 0 0.25 0.5 0.75 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from eval_sap import iou_matrix, greedy_tp, voc_ap, IOU_THRS, load_gt  # noqa: E402

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
N_CLS = 8


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------- AP(与 eval_sap 同构)

def global_ap(preds, gt, frame_ids, iou_thrs=IOU_THRS):
    """preds: list[(box,score,cls)]  frame_ids: list[str]  gt: fn->(box,cls)

    与 eval_sap.evaluate 的积分逻辑一致: 全局按分数降序 -> 按(帧,类)贪心匹配
    -> 逐类累计 -> 逐类 AP -> 平均(跳过无 GT 的类)。
    """
    boxes, scores, clss = [], [], []
    for b, s, c in preds:
        if len(b):
            boxes.append(b)
            scores.append(s)
            clss.append(c)
    if not boxes:
        return 0.0, 0.0, [0.0] * len(iou_thrs)

    box_a = np.concatenate(boxes)
    score_a = np.concatenate(scores)
    cls_a = np.concatenate(clss)
    fid = np.concatenate([np.full(len(b), i, np.int64)
                          for i, (b, s, c) in enumerate(preds) if len(b)])

    order = np.argsort(-score_a, kind='stable')
    box_a, score_a, cls_a, fid = (box_a[order], score_a[order],
                                  cls_a[order], fid[order])
    key = fid * N_CLS + cls_a
    gorder = np.argsort(key, kind='stable')
    sk = key[gorder]
    uniq, starts = np.unique(sk, return_index=True)
    ends = np.append(starts[1:], len(sk))

    # GT 分母: 只统计参与评测的帧
    n_gt = np.zeros(N_CLS, np.int64)
    for i in frame_ids:
        _, gc = gt[i]
        if len(gc):
            n_gt += np.bincount(gc, minlength=N_CLS)

    # 预取每组的 GT, 避免阈值循环里重复查表
    grp = []
    for gi in range(len(uniq)):
        pos = gorder[starts[gi]:ends[gi]]
        u = int(uniq[gi])
        c = u % N_CLS
        fn = frame_ids[u // N_CLS]
        gb, gc = gt[fn]
        m = gc == c
        grp.append((pos, c, gb[m] if m.any() else None))

    cls_pos = [np.nonzero(cls_a == c)[0] for c in range(N_CLS)]
    aps = []
    for thr in iou_thrs:
        tp = np.zeros(len(box_a), bool)
        for pos, c, gts_c in grp:
            if gts_c is None:
                continue
            tp[pos] = greedy_tp(iou_matrix(box_a[pos], gts_c), thr)
        per_cls = []
        for c in range(N_CLS):
            if n_gt[c] == 0:
                continue
            idx = cls_pos[c]
            if len(idx) == 0:
                per_cls.append(0.0)
                continue
            csum = np.cumsum(tp[idx])
            rec = csum / n_gt[c]
            prec = csum / np.arange(1, len(idx) + 1)
            per_cls.append(voc_ap(rec, prec))
        aps.append(float(np.mean(per_cls)) if per_cls else 0.0)
    aps = np.asarray(aps)
    return float(aps[0]), float(aps.mean()), aps.tolist()


def recall_at(per_frame_pred, frame_ids, gt, thr=0.5):
    """recall@IoU=thr: 被至少一个预测框命中的 GT 占比(与分数排序无关)"""
    hit = tot = 0
    for p, fn in zip(per_frame_pred, frame_ids):
        b = p[0]
        gb, gc = gt[fn]
        if len(gb) == 0:
            continue
        tot += len(gb)
        if len(b) == 0:
            continue
        iou = iou_matrix(b, gb)
        hit += int((iou.max(axis=0) >= thr).sum())
    return 100.0 * hit / max(tot, 1)


# ---------------------------------------------------------------- 主体

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, default=6, help='用哪个深度的录制做源预测')
    ap.add_argument('--alphas', nargs='+', type=float, default=[0.0, 0.5, 1.0],
                    help='运动外推的阻尼系数: box_t = box_{t-1} + a*(box_{t-1}-box_{t-2})')
    ap.add_argument('--match-iou', type=float, default=0.3,
                    help='估计运动矢量时, t-1 与 t-2 框的匹配阈值')
    ap.add_argument('--preds-dir', default='stream_preds')
    ap.add_argument('--ann', default=os.path.join(DATA, 'annotations_val.json'))
    ap.add_argument('--stream-index', default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--out', default='out_probe_prop.json')
    args = ap.parse_args()

    with open(args.stream_index, encoding='utf-8') as f:
        idx = json.load(f)
    gt = load_gt(args.ann)
    log(f'GT {sum(len(v[0]) for v in gt.values())} 框, {len(idx["scenes"])} 场景')

    z = np.load(os.path.join(args.preds_dir, f'preds_d{args.depth}.npz'),
                allow_pickle=False)
    rec = {k: z[k] for k in z.files}
    log(f'源预测: depth={args.depth}, {len(rec["scene"])} 帧')

    # 逐帧预测查表: (scene, frame_idx) -> (box, score, cls)
    per_frame = {}
    for i in range(len(rec['scene'])):
        s, e = int(rec['ptr'][i]), int(rec['ptr'][i + 1])
        per_frame[(int(rec['scene'][i]), int(rec['frame_idx'][i]))] = (
            rec['box'][s:e], rec['score'][s:e], rec['cls'][s:e])

    # 构造评测帧集: 每场景内 idx>=2
    targets = []
    for sc in idx['scenes']:
        sid = sc['scene_id']
        for fr in sc['frames']:
            if fr['idx'] >= 2 and (sid, fr['idx']) in per_frame:
                targets.append((sid, fr['idx'], fr['file']))
    frame_ids = [t[2] for t in targets]
    log(f'评测帧: {len(targets)} (每场景 idx>=2)')

    # ---------- model 探针 ----------
    log('')
    log('[1/4] model 探针(模型自己在 t 帧的预测)')
    t0 = time.time()
    preds_model = [per_frame[(s, i)] for s, i, _ in targets]
    m50, m, _ = global_ap(preds_model, gt, frame_ids)
    r50 = recall_at(preds_model, frame_ids, gt)
    log(f'   mAP50={m50:.4f}  mAP50-95={m:.4f}  recall@50={r50:.1f}%  ({time.time()-t0:.0f}s)')

    # ---------- copy 探针 ----------
    log('[2/4] copy 探针(t-1 的框原样当作 t 的预测)')
    t0 = time.time()
    preds_copy = []
    for s, i, _ in targets:
        b, sc, c = per_frame[(s, i - 1)]
        preds_copy.append((b, sc, c))
    c50, c95, _ = global_ap(preds_copy, gt, frame_ids)
    cr50 = recall_at(preds_copy, frame_ids, gt)
    log(f'   mAP50={c50:.4f}  mAP50-95={c95:.4f}  recall@50={cr50:.1f}%  ({time.time()-t0:.0f}s)')

    # ---------- extrap 探针(含 alpha 扫描) ----------
    log('[3/4] extrap 探针(t-2->t-1 运动矢量外推)')
    # 先算好每帧的运动矢量(独立于 alpha)
    vel_of = {}
    n_match_tot = n_box_tot = 0
    t0 = time.time()
    for s, i, _ in targets:
        b1, _, c1 = per_frame[(s, i - 1)]
        b2, _, c2 = per_frame[(s, i - 2)]
        v = np.zeros_like(b1)
        if len(b1) and len(b2):
            iou = iou_matrix(b1, b2)
            same = (c1[:, None] == c2[None, :])
            iou = np.where(same, iou, -1.0)
            best = iou.argmax(axis=1)
            bi = iou[np.arange(len(b1)), best]
            ok = bi >= args.match_iou
            v[ok] = b1[ok] - b2[best[ok]]
            n_match_tot += int(ok.sum())
        n_box_tot += len(b1)
        vel_of[(s, i)] = v
    log(f'   运动矢量估计: {n_match_tot}/{n_box_tot} 个框匹配上 '
        f'({100*n_match_tot/max(n_box_tot,1):.1f}%)  ({time.time()-t0:.0f}s)')

    results = {}
    log('')
    log(f"   {'alpha':>7}{'mAP50':>10}{'mAP50-95':>11}{'recall@50':>11}")
    for a in args.alphas:
        t0 = time.time()
        preds = []
        for s, i, _ in targets:
            b, sc, c = per_frame[(s, i - 1)]
            if a > 0 and len(b):
                b = b + a * vel_of[(s, i)]
            preds.append((b, sc, c))
        a50, a95, _ = global_ap(preds, gt, frame_ids)
        ar50 = recall_at(preds, frame_ids, gt)
        results[a] = dict(map50=a50, map=a95, recall=ar50)
        log(f'   {a:>7.2f}{a50:>10.4f}{a95:>11.4f}{ar50:>11.1f}%  ({time.time()-t0:.0f}s)')

    # ---------- 汇总 ----------
    best_a = max(results, key=lambda k: results[k]['map50'])
    gap = m50 - results[best_a]['map50']
    log('')
    log('=' * 68)
    log(f"{'探针':<26}{'mAP50':>10}{'mAP50-95':>11}{'recall@50':>11}")
    log(f"{'model (自己看 t 帧)':<26}{m50:>10.4f}{m:>11.4f}{r50:>10.1f}%")
    log(f"{'copy (零阶保持)':<26}{c50:>10.4f}{c95:>11.4f}{cr50:>10.1f}%")
    for a in args.alphas:
        nm = 'extrap a=%.2f' % a + ('  (最右)' if a == best_a else '')
        log(f"{nm:<26}{results[a]['map50']:>10.4f}{results[a]['map']:>11.4f}"
            f"{results[a]['recall']:>10.1f}%")
    log('=' * 68)
    log(f'最佳外推 alpha={best_a}: 相对模型自身差 {gap:.4f} mAP50')
    log(f'  即"只靠前帧位置信息"能达到模型自身水平的 '
        f'{100*results[best_a]["map50"]/max(m50,1e-9):.1f}%')
    log('')
    if results[best_a]['map50'] >= 0.25:
        log('判定: 前帧位置信息量充足 -> 锚点层传播(方案①)值得先试')
    elif results[best_a]['map50'] >= 0.12:
        log('判定: 中等 -> 锚点传播可作起点, 但需配合内容嵌入(方案②)')
    else:
        log('判定: 位置信息不足 -> 必须依赖内容/记忆库, 工程难度较大')

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(dict(depth=args.depth, alphas=args.alphas,
                       model=dict(map50=m50, map=m, recall=r50),
                       copy=dict(map50=c50, map=c95, recall=cr50),
                       extrap={str(k): v for k, v in results.items()},
                       match_rate_pct=100 * n_match_tot / max(n_box_tot, 1),
                       n_frames=len(targets)),
                  f, ensure_ascii=False, indent=2)
    log(f'已写入 {args.out}')


if __name__ == '__main__':
    main()
