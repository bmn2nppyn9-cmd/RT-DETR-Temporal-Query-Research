# -*- coding: utf-8 -*-
"""Zero-training offline feasibility analysis for Path B query propagation.

Uses already-recorded depth-3/depth-6 predictions. It measures:
1) whether propagated depth-3 boxes rescue GT missed by current depth-3 boxes;
2) class-agnostic and class-aware geometric recall for M propagated slots;
3) a conservative lower bound from score-decayed offline fusion (no decoder refinement).

This cannot simulate cross-attention refinement of propagated hidden queries. The geometric
union is an upper-bound/headroom test; score fusion is a lower-bound sanity check.
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
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from eval_sap import iou_matrix, load_gt
from probe_propagation import global_ap

DATA = r"D:\lunwen\自研\数据集\argoverse-hd"


def log(msg):
    print(msg, flush=True)


def load_record(path):
    z = np.load(path, allow_pickle=False)
    rec = {k: z[k] for k in z.files}
    per = {}
    for i in range(len(rec["scene"])):
        s, e = int(rec["ptr"][i]), int(rec["ptr"][i + 1])
        per[(int(rec["scene"][i]), int(rec["frame_idx"][i]))] = (
            rec["box"][s:e], rec["score"][s:e], rec["cls"][s:e]
        )
    return rec, per


def dedup_positions(box, score, cls):
    """Collapse postprocess's repeated (same query box, different class) entries."""
    if len(box) == 0:
        return box, score, cls
    order = np.argsort(-score, kind="stable")
    b, s, c = box[order], score[order], cls[order]
    _, first = np.unique(np.round(b, 1), axis=0, return_index=True)
    first = np.sort(first)
    return b[first], s[first], c[first]


def extrapolate(prev1, prev2, alpha, match_iou, top_m):
    b1, s1, c1 = dedup_positions(*prev1)
    b2, _s2, _c2 = dedup_positions(*prev2)
    if len(b1) and len(b2) and alpha != 0:
        iou = iou_matrix(b1, b2)  # class agnostic: class must be re-estimated later
        best = iou.argmax(axis=1)
        ok = iou[np.arange(len(b1)), best] >= match_iou
        velocity = np.zeros_like(b1)
        velocity[ok] = b1[ok] - b2[best[ok]]
        b1 = b1 + alpha * velocity
    if len(b1):
        b1[:, [0, 2]] = np.clip(b1[:, [0, 2]], 0, 1920)
        b1[:, [1, 3]] = np.clip(b1[:, [1, 3]], 0, 1200)
    return b1[:top_m], s1[:top_m], c1[:top_m]


def hit_mask(pred_box, pred_cls, gt_box, gt_cls, thr, class_aware=False):
    if len(gt_box) == 0:
        return np.zeros(0, dtype=bool)
    if len(pred_box) == 0:
        return np.zeros(len(gt_box), dtype=bool)
    iou = iou_matrix(pred_box, gt_box)
    if class_aware:
        iou = np.where(pred_cls[:, None] == gt_cls[None, :], iou, -1.0)
    return iou.max(axis=0) >= thr


def fuse(cur, prop, gamma, max_det, novelty_iou=None):
    cb, cs, cc = cur
    pb, ps, pc = prop
    if novelty_iou is not None and len(pb) and len(cb):
        cur_unique, _s, _c = dedup_positions(cb, cs, cc)
        keep = iou_matrix(pb, cur_unique).max(axis=1) < novelty_iou
        pb, ps, pc = pb[keep], ps[keep], pc[keep]
    if len(pb):
        b = np.concatenate([cb, pb])
        s = np.concatenate([cs, ps * gamma])
        c = np.concatenate([cc, pc])
    else:
        b, s, c = cb, cs, cc
    order = np.argsort(-s, kind="stable")[:max_det]
    return b[order], s[order], c[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", default="stream_preds")
    ap.add_argument("--ann", default=os.path.join(DATA, "annotations_val.json"))
    ap.add_argument("--stream-index", default=os.path.join(DATA, "stream_index.json"))
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--match-iou", type=float, default=0.3)
    ap.add_argument("--slots", nargs="+", type=int, default=[25, 50, 75, 150, 300])
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--fusion-slots", type=int, default=75)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--out", default="out_offline_query_prop.json")
    args = ap.parse_args()

    _r3, p3 = load_record(os.path.join(args.preds_dir, "preds_d3.npz"))
    _r6, p6 = load_record(os.path.join(args.preds_dir, "preds_d6.npz"))
    with open(args.stream_index, encoding="utf-8") as f:
        stream = json.load(f)
    gt = load_gt(args.ann)

    targets = []
    for scene in stream["scenes"]:
        sid = int(scene["scene_id"])
        for fr in scene["frames"]:
            fi = int(fr["idx"])
            key = (sid, fi)
            if fi >= 2 and key in p3 and key in p6 and (sid, fi - 1) in p3 and (sid, fi - 2) in p3:
                targets.append((sid, fi, fr["file"]))
    frame_ids = [x[2] for x in targets]
    log(f"评测帧: {len(targets)}")

    thresholds = (0.3, 0.5, 0.7)
    modes = ("agnostic", "class_aware")
    geometry = {}
    per_class = {}
    t0 = time.time()

    # Cache per-frame current/teacher/prop data once.
    cached = []
    for n, (sid, fi, fn) in enumerate(targets):
        cur_raw = p3[(sid, fi)]
        d6_raw = p6[(sid, fi)]
        cur = dedup_positions(*cur_raw)
        d6 = dedup_positions(*d6_raw)
        props = {
            m: extrapolate(p3[(sid, fi - 1)], p3[(sid, fi - 2)], args.alpha, args.match_iou, m)
            for m in args.slots
        }
        cached.append((fn, cur_raw, cur, d6, props))
        if (n + 1) % 3000 == 0:
            log(f"  cache {n+1}/{len(targets)}")

    for m in args.slots:
        geometry[str(m)] = {}
        for mode in modes:
            aware = mode == "class_aware"
            stat = {
                thr: {k: 0 for k in ("gt", "current", "prop", "union", "d6", "rescued")}
                for thr in thresholds
            }
            pc = {
                c: {k: 0 for k in ("gt", "current", "prop", "union", "d6", "rescued")}
                for c in range(8)
            }
            for fn, _cur_raw, cur, d6, props in cached:
                gb, gc = gt[fn]
                pb, ps, pcl = props[m]
                cb, cs, ccl = cur
                db, ds, dcl = d6
                for thr in thresholds:
                    hc = hit_mask(cb, ccl, gb, gc, thr, aware)
                    hp = hit_mask(pb, pcl, gb, gc, thr, aware)
                    hd = hit_mask(db, dcl, gb, gc, thr, aware)
                    u = hc | hp
                    q = stat[thr]
                    q["gt"] += len(gb)
                    q["current"] += int(hc.sum())
                    q["prop"] += int(hp.sum())
                    q["union"] += int(u.sum())
                    q["d6"] += int(hd.sum())
                    q["rescued"] += int((~hc & hp).sum())
                    if thr == 0.5:
                        for c in range(8):
                            mask = gc == c
                            z = pc[c]
                            z["gt"] += int(mask.sum())
                            z["current"] += int(hc[mask].sum())
                            z["prop"] += int(hp[mask].sum())
                            z["union"] += int(u[mask].sum())
                            z["d6"] += int(hd[mask].sum())
                            z["rescued"] += int((~hc[mask] & hp[mask]).sum())
            geometry[str(m)][mode] = {
                str(thr): {
                    "gt": q["gt"],
                    "current_recall": 100 * q["current"] / max(q["gt"], 1),
                    "prop_recall": 100 * q["prop"] / max(q["gt"], 1),
                    "union_recall": 100 * q["union"] / max(q["gt"], 1),
                    "d6_recall": 100 * q["d6"] / max(q["gt"], 1),
                    "rescued_gt": q["rescued"],
                    "rescue_pp": 100 * q["rescued"] / max(q["gt"], 1),
                    "miss_rescue_rate": 100 * q["rescued"] / max(q["gt"] - q["current"], 1),
                }
                for thr, q in stat.items()
            }
            if m == args.fusion_slots:
                per_class[mode] = {
                    str(c): {
                        "gt": z["gt"],
                        "current_recall": 100 * z["current"] / max(z["gt"], 1),
                        "prop_recall": 100 * z["prop"] / max(z["gt"], 1),
                        "union_recall": 100 * z["union"] / max(z["gt"], 1),
                        "d6_recall": 100 * z["d6"] / max(z["gt"], 1),
                        "rescue_pp": 100 * z["rescued"] / max(z["gt"], 1),
                    }
                    for c, z in pc.items()
                }

    log(f"几何分析完成 ({time.time()-t0:.1f}s)")
    log("")
    log("几何上限（类无关）")
    log(f"{'M':>5}{'IoU':>7}{'当前3层':>10}{'传播':>10}{'并集':>10}{'6层':>10}{'救回pp':>10}{'漏检救回%':>12}")
    for m in args.slots:
        for thr in thresholds:
            x = geometry[str(m)]["agnostic"][str(thr)]
            log(f"{m:>5}{thr:>7.1f}{x['current_recall']:>9.2f}%{x['prop_recall']:>9.2f}%"
                f"{x['union_recall']:>9.2f}%{x['d6_recall']:>9.2f}%{x['rescue_pp']:>9.2f}"
                f"{x['miss_rescue_rate']:>11.2f}%")

    # Conservative AP lower bound: current d3 outputs + propagated d3 outputs, no refinement.
    fusion = {}
    m = args.fusion_slots
    for novelty_name, novelty_iou in (("raw", None), ("novel_iou0.7", 0.7)):
        fusion[novelty_name] = {}
        for gamma in args.gammas:
            preds = []
            for _fn, cur_raw, _cur, _d6, props in cached:
                preds.append(fuse(cur_raw, props[m], gamma, args.max_det, novelty_iou))
            f50, fmap, _ = global_ap(preds, gt, frame_ids)
            fusion[novelty_name][str(gamma)] = {"map50": f50, "map": fmap}
            log(f"融合 M={m} {novelty_name} gamma={gamma:.2f}: mAP50={f50:.4f} mAP={fmap:.4f}")

    # Same-frame baselines on the identical target set.
    pred3 = [p3[(sid, fi)] for sid, fi, _fn in targets]
    pred6 = [p6[(sid, fi)] for sid, fi, _fn in targets]
    b350, b3map, _ = global_ap(pred3, gt, frame_ids)
    b650, b6map, _ = global_ap(pred6, gt, frame_ids)
    baselines = {
        "d3": {"map50": b350, "map": b3map},
        "d6": {"map50": b650, "map": b6map},
    }
    log(f"基线 d3: mAP50={b350:.4f} mAP={b3map:.4f}")
    log(f"基线 d6: mAP50={b650:.4f} mAP={b6map:.4f}")

    out = {
        "n_frames": len(targets),
        "alpha": args.alpha,
        "match_iou": args.match_iou,
        "slots": args.slots,
        "geometry": geometry,
        "per_class_at_iou_0.5_for_fusion_slots": per_class,
        "fusion_slots": m,
        "fusion": fusion,
        "baselines": baselines,
        "interpretation_boundary": {
            "geometry": "headroom/necessary-condition; no decoder refinement or score ranking",
            "fusion": "conservative lower bound using stale boxes/classes/scores; not equivalent to query propagation",
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
