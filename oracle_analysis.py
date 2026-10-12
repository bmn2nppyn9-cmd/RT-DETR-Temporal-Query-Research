# -*- coding: utf-8 -*-
"""
oracle_analysis.py —— 自适应解码深度的上界分析
==============================================
回答一个问题: 逐帧自适应选择解码深度, 最多能赚多少?

三个层次, 分别对应三种"值不值得做"的判断:

  A. 精度上界(逐帧最优)
     每帧各自选 AP 最高的深度, 汇总后算全局 mAP, 与最优静态深度比较。
     -> 这是"完美的按帧路由器"能拿到的精度, 也是路径 A 的精度天花板。

  B. 效率上界(最浅够用)
     每帧选"最浅且精度不掉"的深度(可设容差)。
     -> 这是"只要不掉精度, 能省多少计算"的天花板。

  C. 流式上界(sAP)
     用混合后的实际推理耗时驱动流式调度, 在各注入延迟下与最优静态比较。
     -> 这是最贴近论文叙事的一个数: 高延迟下 oracle 因为少丢帧而更稳。

判据(沿用项目既有标准):
    增益 < 1.0 pp  -> 空间不足, 不建议做路由器
    1.0 ~ 2.0 pp  -> 空间有限, 降为次要贡献
    > 2.0 pp      -> 成立, 继续做真实路由器

重要前提: 本文件依赖 stream_preds/ 下的三个录制文件, 且
eval_sap.py 的 GT 分母必须已修正为"评测范围内所有帧"(否则丢帧免费, 结论会反)。

用法:
    python oracle_analysis.py
    python oracle_analysis.py --sweep 0 10 15 20 30
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

from eval_sap import (build_timeline, load_gt, evaluate, iou_matrix,  # noqa: E402
                      greedy_tp, voc_ap)

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
DEPTHS = [1, 3, 6]


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------- 逐帧打分

def per_frame_ap(box, score, cls, gb, gc, thr=0.5):
    """单帧的类平均 AP。返回 None 表示该帧无 GT(无法评判)。"""
    if len(gb) == 0:
        return None
    if len(box) == 0:
        return 0.0
    o = np.argsort(-score, kind='stable')
    box, cls = box[o], cls[o]
    aps = []
    for c in np.unique(gc):
        m = gc == c
        dt = cls == c
        if not dt.any():
            aps.append(0.0)
            continue
        tp = greedy_tp(iou_matrix(box[dt], gb[m]), thr)
        csum = np.cumsum(tp)
        rec = csum / float(m.sum())
        prec = csum / np.arange(1, len(tp) + 1)
        aps.append(voc_ap(rec, prec))
    return float(np.mean(aps)) if aps else 0.0


def build_mixed(recs, assign):
    """按 assign(每帧选中的深度) 拼出一个混合 rec"""
    boxes, scores, clss = [], [], []
    ptr = np.zeros(len(assign) + 1, np.int64)
    comp = np.zeros(len(assign), np.float32)
    for i, d in enumerate(assign):
        r = recs[int(d)]
        s, e = int(r['ptr'][i]), int(r['ptr'][i + 1])
        boxes.append(r['box'][s:e])
        scores.append(r['score'][s:e])
        clss.append(r['cls'][s:e])
        ptr[i + 1] = ptr[i] + (e - s)
        comp[i] = r['compute_ms'][i]
    base = recs[int(assign[0])]
    return dict(
        box=np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
        score=np.concatenate(scores) if scores else np.zeros(0, np.float32),
        cls=np.concatenate(clss) if clss else np.zeros(0, np.uint8),
        ptr=ptr.astype(np.int32), compute_ms=comp,
        scene=base['scene'], frame_idx=base['frame_idx'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds-dir', default='stream_preds')
    ap.add_argument('--ann', default=os.path.join(DATA, 'annotations_val.json'))
    ap.add_argument('--stream-index', default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--static-json', default='out_sap.json',
                    help='静态深度的 sAP 扫描结果(用于对比)')
    ap.add_argument('--sweep', nargs='+', type=float, default=[0, 10, 15, 20, 30])
    ap.add_argument('--sel-thr', type=float, default=0.5,
                    help='逐帧选择的 IoU 阈值(选 AP@该阈值 最大的深度)')
    ap.add_argument('--out', default='out_oracle.json')
    args = ap.parse_args()

    frames, scene_t, scene_files = build_timeline(args.stream_index)
    frame_key = {(f[0], f[1]): i for i, f in enumerate(frames)}
    gt = load_gt(args.ann)
    nf = len(frames)
    log(f'时序 {nf} 帧 / {len(scene_t)} 场景,  GT {sum(len(v[0]) for v in gt.values())} 框')

    recs = {}
    for d in DEPTHS:
        p = os.path.join(args.preds_dir, f'preds_d{d}.npz')
        if not os.path.exists(p):
            log(f'[错误] 缺 {p}')
            sys.exit(1)
        z = np.load(p, allow_pickle=False)
        recs[d] = {k: z[k] for k in z.files}
    log(f'已载入深度 {DEPTHS} 的录制')

    # ---------- 逐帧打分 ----------
    log('')
    log('[1/3] 逐帧打分(每帧×每深度 算一次类平均 AP@%.2f)...' % args.sel_thr)
    t0 = time.time()
    # 帧 -> 该帧 GT
    gt_of = {}
    for s_id, fns in scene_files.items():
        for fn in fns:
            gt_of[fn] = gt[fn]

    score_tab = np.full((nf, len(DEPTHS)), np.nan, np.float64)
    for j, d in enumerate(DEPTHS):
        r = recs[d]
        for i in range(nf):
            fn = scene_files[int(r['scene'][i])][int(r['frame_idx'][i])]
            gb, gc = gt_of[fn]
            s, e = int(r['ptr'][i]), int(r['ptr'][i + 1])
            score_tab[i, j] = np.nan if len(gb) == 0 else \
                per_frame_ap(r['box'][s:e], r['score'][s:e], r['cls'][s:e], gb, gc,
                             args.sel_thr)
        log(f'   depth={d} 完成  ({time.time()-t0:.0f}s)')
    log(f'   打分耗时 {time.time()-t0:.0f}s')

    valid = ~np.isnan(score_tab).any(axis=1)      # 有 GT 的帧
    n_valid = int(valid.sum())
    log(f'   有效帧 {n_valid} / {nf} (其余无 GT)')

    # ---------- A. 精度上界 ----------
    log('')
    log('[2/3] A. 精度上界(逐帧选最优)')
    # 同分时取更浅的深度(省算力)
    best_j = np.zeros(nf, np.int64)
    for i in range(nf):
        if not valid[i]:
            best_j[i] = 0
            continue
        row = score_tab[i]
        m = np.nanmax(row)
        cand = np.nonzero(row >= m - 1e-9)[0]
        best_j[i] = cand.min()
    assign = np.array([DEPTHS[j] for j in best_j], np.int64)

    cnt = {d: int((assign == d).sum()) for d in DEPTHS}
    log('   逐帧最优深度的分布:')
    for d in DEPTHS:
        log(f'      {d} 层: {cnt[d]:>6} 帧 ({100*cnt[d]/nf:5.1f}%)')

    mixed = build_mixed(recs, assign)
    # 静态各深度的纯 AP(verify=True, 全部帧被处理)
    ap_stat = {}
    for d in DEPTHS:
        r = evaluate(recs[d], gt, frames, scene_t, scene_files, 0.0,
                     verify=True, frame_key=frame_key)
        ap_stat[d] = r
    ap_oracle = evaluate(mixed, gt, frames, scene_t, scene_files, 0.0,
                         verify=True, frame_key=frame_key)

    log('')
    log(f"   {'方式':<18}{'mAP50':>10}{'mAP50-95':>12}")
    for d in DEPTHS:
        log(f"   {'全程 '+str(d)+' 层':<18}{ap_stat[d]['sap50']:>10.4f}{ap_stat[d]['sap']:>12.4f}")
    log(f"   {'逐帧 oracle':<18}{ap_oracle['sap50']:>10.4f}{ap_oracle['sap']:>12.4f}")
    best_static50 = max(ap_stat[d]['sap50'] for d in DEPTHS)
    best_static95 = max(ap_stat[d]['sap'] for d in DEPTHS)
    gain50 = (ap_oracle['sap50'] - best_static50) * 100
    gain95 = (ap_oracle['sap'] - best_static95) * 100
    log(f"   增益 vs 最优静态:  mAP50 {gain50:+.2f} pp   mAP50-95 {gain95:+.2f} pp")

    # ---------- B. 效率上界 ----------
    log('')
    log('   效率上界: 每帧选"最浅且够用"的深度')
    log(f"   {'容差':<8}{'1层':>8}{'3层':>8}{'6层':>8}{'平均深度':>10}{'相对6层省时':>13}")
    eff = {}
    comp = {d: float(np.median(recs[d]['compute_ms'])) for d in DEPTHS}
    for tol in (0.0, 0.01, 0.05, 0.10):
        cnt2 = {d: 0 for d in DEPTHS}
        depths_chosen = []
        for i in range(nf):
            if not valid[i]:
                cnt2[1] += 1
                depths_chosen.append(1)
                continue
            row = score_tab[i]
            ref = row[2]                       # 6 层为参照
            if ref <= 0:
                cnt2[1] += 1
                depths_chosen.append(1)        # 6 层也检不出东西 -> 用最浅
                continue
            ch = 6
            for j, d in enumerate(DEPTHS):
                if row[j] >= (1.0 - tol) * ref:
                    ch = d
                    break
            cnt2[ch] += 1
            depths_chosen.append(ch)
        depths_chosen = np.array(depths_chosen)
        mean_d = float(np.mean(depths_chosen))
        saving = float(np.mean([1.0 - comp[int(d)] / comp[6] for d in depths_chosen]) * 100)
        eff[tol] = dict(counts=cnt2, mean_depth=mean_d, saving_pct=saving)
        log(f"   {('%.0f%%' % (tol*100)):<8}{cnt2[1]:>8}{cnt2[3]:>8}{cnt2[6]:>8}"
            f"{mean_d:>10.2f}{saving:>12.1f}%")

    # ---------- C. 流式上界 ----------
    log('')
    log('[3/3] C. 流式(sAP)上界')
    static = None
    if os.path.exists(args.static_json):
        static = json.load(open(args.static_json, encoding='utf-8'))
    log(f"   {'注入L':>7}{'最优静态 sAP50':>18}{'oracle sAP50':>15}{'增益(pp)':>11}"
        f"{'静态丢弃':>10}{'oracle丢弃':>12}")
    sap_rows = []
    for L in args.sweep:
        r_or = evaluate(mixed, gt, frames, scene_t, scene_files, L,
                        verify=False, frame_key=frame_key)
        if static:
            cand = []
            for k, lst in static['results'].items():
                for row in lst:
                    if abs(row['added_ms'] - L) < 1e-6:
                        cand.append(row['sap50'])
            st50 = max(cand) if cand else float('nan')
            st_drop = min([row['drop_pct'] for k, lst in static['results'].items()
                           for row in lst if abs(row['added_ms'] - L) < 1e-6]
                          or [float('nan')])
        else:
            st50, st_drop = float('nan'), float('nan')
        sap_rows.append(dict(L=L, oracle=r_or['sap50'], static=st50,
                             gain=(r_or['sap50'] - st50) * 100 if st50 == st50 else None,
                             drop=r_or['drop_pct']))
        log(f"   {L:>7.0f}{st50:>18.4f}{r_or['sap50']:>15.4f}"
            f"{(r_or['sap50']-st50)*100 if st50==st50 else float('nan'):>11.2f}"
            f"{st_drop:>9.1f}%{r_or['drop_pct']:>11.1f}%")

    # ---------- 判定 ----------
    log('')
    log('=' * 66)
    log('判定(项目既有标准: >2.0pp 成立 / 1~2pp 空间有限 / <1pp 不建议)')
    log('=' * 66)
    def verdict(g, name):
        if g >= 2.0:
            v = '成立, 值得做真实路由器'
        elif g >= 1.0:
            v = '空间有限, 建议降为次要贡献'
        else:
            v = '空间不足, 不建议投入路由器'
        log(f'  {name:<22} {g:+.2f} pp  -> {v}')
    verdict(gain50, '精度上界(vs 最优静态)')
    if eff.get(0.01):
        log(f'  {"效率上界(1%容差)":<22} 省时 {eff[0.01]["saving_pct"]:.1f}%  '
            f'平均深度 {eff[0.01]["mean_depth"]:.2f}')
    gains_sap = [r['gain'] for r in sap_rows if r['gain'] is not None]
    if gains_sap:
        log(f'  {"流式上界(最大增益)":<22} {max(gains_sap):+.2f} pp')

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(dict(
            depth_counts=cnt,
            static_ap={str(d): ap_stat[d] for d in DEPTHS},
            oracle_ap=ap_oracle,
            gain50=gain50, gain95=gain95,
            efficiency={str(k): v for k, v in eff.items()},
            compute_median={str(d): comp[d] for d in DEPTHS},
            sap=sap_rows,
        ), f, ensure_ascii=False, indent=2, default=float)
    log('')
    log(f'已写入 {args.out}')


if __name__ == '__main__':
    main()
