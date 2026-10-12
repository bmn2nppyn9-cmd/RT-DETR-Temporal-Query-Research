# -*- coding: utf-8 -*-
"""diag_b0_posthoc.py —— B0 机理分析 (纯离线, 不改任何模型输出)
=====================================================
基于已录制的 a0/b0 npz, 回答 "B0 为什么有效/无效":
  D1 重复框过滤诊断: 只抑制 temporal 中 "与更高分同类检测 IoU>=thr 重复" 的条目
      (current 条目永不被抑制), 重算 AP —— 回答 "去重后 B0 能否回到 A0 以上"
  D2 score 校准诊断: temporal 分数乘 gamma 后重算 AP —— 回答 "是否单纯分数偏高"
  D3 新颖性上限: 丢弃所有与 current 重叠 (IoU>=0.5, 类无关) 的 temporal 条目,
      重算 AP —— 回答 "temporal 独立贡献的上限"
  D4 干扰分析: A0 高分检测在 B0 的 current 来源检测中的存活率与分数漂移 ——
      回答 "temporal query 是否通过 self-attention 损害了 current query"

所有分析都是录制后离线进行, 不影响四组主实验的任何输出。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, r'D:\lunwen\自研')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from eval_sap import build_timeline, load_gt
from eval_b0 import evaluate_ap

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'


def log(m):
    print(m, flush=True)


def iou_xyxy(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    ab = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.maximum(aa + ab - inter, 1e-9)


def repack(box, score, cls, src, nf):
    """把逐帧列表重打包成 evaluate_ap 需要的 rec 结构"""
    ptr = [0]
    bb, ss, cc, rr = [], [], [], []
    for i in range(nf):
        bb.append(box[i])
        ss.append(score[i])
        cc.append(cls[i])
        rr.append(src[i])
        ptr.append(ptr[-1] + len(box[i]))
    return {
        'scene': np.arange(nf, dtype=np.int16),
        'box': np.concatenate(bb) if bb else np.zeros((0, 4), np.float32),
        'score': np.concatenate(ss) if ss else np.zeros(0, np.float32),
        'cls': np.concatenate(cc) if cc else np.zeros(0, np.uint8),
        'src': np.concatenate(rr) if rr else np.zeros(0, np.int8),
        'ptr': np.array(ptr, np.int32),
    }


def per_frame(rec):
    nf = len(rec['scene'])
    out = []
    for i in range(nf):
        s, e = int(rec['ptr'][i]), int(rec['ptr'][i + 1])
        o = np.argsort(-rec['score'][s:e], kind='stable')
        out.append((rec['box'][s:e][o], rec['score'][s:e][o],
                    rec['cls'][s:e][o], rec['src'][s:e][o]))
    return out


def d1_dedup(frames, thr):
    """只抑制 temporal 重复条目: 与任一更高分【同类】检测 IoU>=thr"""
    out = []
    for b, s, c, r in frames:
        keep = np.ones(len(b), bool)
        for d in range(len(b)):
            if r[d] != 1:
                continue
            higher = (s > s[d]) | ((s == s[d]) & (np.arange(len(b)) < d))
            same = higher & (c == c[d]) & keep
            if same.any() and iou_xyxy(b[d:d + 1], b[same]).max() >= thr:
                keep[d] = False
        out.append((b[keep], s[keep], c[keep], r[keep]))
    return out


def d2_gamma(frames, gamma):
    out = []
    for b, s, c, r in frames:
        s2 = np.where(r == 1, s * gamma, s).astype(np.float32)
        o = np.argsort(-s2, kind='stable')
        out.append((b[o], s2[o], c[o], r[o]))
    return out


def d3_novelty(frames, thr):
    out = []
    for b, s, c, r in frames:
        tm = r == 1
        cm = r == 0
        keep = np.ones(len(b), bool)
        if tm.any() and cm.any():
            ov = iou_xyxy(b[tm], b[cm]).max(axis=1)
            drop = np.zeros(len(b), bool)
            drop[np.nonzero(tm)[0][ov >= thr]] = True
            keep = ~drop
        out.append((b[keep], s[keep], c[keep], r[keep]))
    return out


def d4_interference(frames_a0, frames_b0, score_thr=0.05):
    n_a, n_hit, n_push = 0, 0, 0
    dscore = []
    for (ba, sa, ca, _ra), (bb, sb, cb, rb) in zip(frames_a0, frames_b0):
        hi = sa > score_thr
        cur = rb == 0
        for d in np.nonzero(hi)[0]:
            n_a += 1
            same = cur & (cb == ca[d])
            if not same.any():
                n_push += 1
                continue
            ov = iou_xyxy(ba[d:d + 1], bb[same])[0]
            j = int(ov.argmax())
            if ov[j] >= 0.95:
                n_hit += 1
                dscore.append(float(sb[same][j] - sa[d]))
    dscore = np.asarray(dscore)
    return {
        'a0_highscore_total': n_a,
        'preserved_in_b0_current': n_hit,
        'missing_in_b0_current': n_a - n_hit,
        'preserve_rate': n_hit / max(n_a, 1),
        'score_shift_mean': float(dscore.mean()) if len(dscore) else 0.0,
        'score_shift_abs_median': float(np.median(np.abs(dscore))) if len(dscore) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds-dir', default=r'D:\lunwen\自研\query_prop_b0\stream_preds')
    ap.add_argument('--ann', default=os.path.join(DATA, 'annotations_val.json'))
    ap.add_argument('--stream-index', default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--suffix', default='smoke')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    frames_tl, _st, _sf = build_timeline(args.stream_index)
    frame_key = {(f[0], f[1]): i for i, f in enumerate(frames_tl)}
    gt = load_gt(args.ann)

    recs = {}
    for cfg in ('a0', 'b0'):
        z = np.load(os.path.join(args.preds_dir, f'preds_{cfg}.npz'), allow_pickle=False)
        recs[cfg] = {k: z[k] for k in z.files}
    rec_b0 = recs['b0']
    nf = len(rec_b0['scene'])
    gidx = [frame_key[(int(rec_b0['scene'][i]), int(rec_b0['frame_idx'][i]))] for i in range(nf)]
    frame_files = [frames_tl[g][3] for g in gidx]

    fa0 = per_frame(recs['a0'])
    fb0 = per_frame(rec_b0)

    def ap_of(frames):
        rec = repack(*zip(*frames), nf) if False else repack(
            [f[0] for f in frames], [f[1] for f in frames],
            [f[2] for f in frames], [f[3] for f in frames], nf)
        r = evaluate_ap(rec, gt, frame_files)
        return r['map50'], r['map'], r['ap75']

    out = {}
    m50, mmap, a75 = ap_of(fa0)
    out['A0'] = dict(map50=m50, map=mmap, ap75=a75)
    log(f'A0:            mAP50={m50:.5f} mAP={mmap:.5f} AP75={a75:.5f}')
    m50, mmap, a75 = ap_of(fb0)
    out['B0_raw'] = dict(map50=m50, map=mmap, ap75=a75)
    log(f'B0 原始:       mAP50={m50:.5f} mAP={mmap:.5f} AP75={a75:.5f}')

    out['D1_dedup'] = {}
    for thr in (0.5, 0.7, 0.9):
        m50, mmap, a75 = ap_of(d1_dedup(fb0, thr))
        out['D1_dedup'][str(thr)] = dict(map50=m50, map=mmap, ap75=a75)
        log(f'D1 去重 IoU>={thr}: mAP50={m50:.5f} mAP={mmap:.5f} AP75={a75:.5f}')

    out['D2_gamma'] = {}
    for g in (0.9, 0.75, 0.5, 0.3, 0.1):
        m50, mmap, a75 = ap_of(d2_gamma(fb0, g))
        out['D2_gamma'][str(g)] = dict(map50=m50, map=mmap, ap75=a75)
        log(f'D2 gamma={g}:  mAP50={m50:.5f} mAP={mmap:.5f} AP75={a75:.5f}')

    out['D3_novelty'] = {}
    for thr in (0.5, 0.3):
        m50, mmap, a75 = ap_of(d3_novelty(fb0, thr))
        out['D3_novelty'][str(thr)] = dict(map50=m50, map=mmap, ap75=a75)
        log(f'D3 只留新颖 IoU<{thr}: mAP50={m50:.5f} mAP={mmap:.5f} AP75={a75:.5f}')

    out['D4_interference'] = d4_interference(fa0, fb0)
    d4 = out['D4_interference']
    log(f"D4 干扰: A0 高分检测在 B0-current 中存活 {d4['preserve_rate']:.4f} "
        f"({d4['preserved_in_b0_current']}/{d4['a0_highscore_total']}), "
        f"分数漂移均值 {d4['score_shift_mean']:+.5f} 中位绝对 {d4['score_shift_abs_median']:.5f}")

    path = args.out or os.path.join(
        r'D:\lunwen\自研\query_prop_b0', f'out_b0_posthoc_{args.suffix}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f'已写入 {path}')


if __name__ == '__main__':
    main()
