# -*- coding: utf-8 -*-
"""
eval_sap.py —— 流式感知 sAP 评测(自实现)
==========================================
sAP 与普通 AP 的唯一区别:
    普通 AP —— 用第 i 帧的预测, 对比第 i 帧的 GT
    sAP     —— 用第 i 帧的预测, 对比【预测产出时刻】对应的 GT

    模型处理第 i 帧需要 T 毫秒, 结果在 t_i + T 时刻才可用。此时传感器可能
    已经前进, 于是要拿"已经变了"的 GT 对答案 —— 这就是流式场景的真实处境。

按【产出时刻所在位置】做 GT 关联, 取"不晚于产出时刻的最后一帧"(floor):
    T < 帧间隔  -> 产出早于下一帧到达, 关联回自己那帧, 无惩罚 (跟上了)
    T > 帧间隔  -> 产出已越过下一帧, 关联到更晚的帧, 开始扣分 (落后了)
  这正是流式感知"跟上就不罚、落后才罚"的语义, 也是 L=0 时能退化为普通 AP
  的原因 —— 因此可用 --verify 自检。

流式调度模型(丢弃式, 真实实时系统的行为):
    处理器一次只处理一帧, 忙到 start + T + L 为止; 期间到达的帧被丢弃。
    T = 该帧实测推理耗时(录制所得),  L = 额外注入的延迟
    (L 模拟推理之外的开销: 图像解码/缩放/NMS/下游环节, 或更慢的硬件)

用法:
    # 1) 自检: AP50 必须等于 ultralytics 报的 mAP50
    python eval_sap.py --verify --depths 1 3 6

    # 2) 真实时序 sAP
    python eval_sap.py --depths 1 3 6

    # 3) 延迟扫描 -> sAP vs 端到端延迟曲线(流式文献的标准图)
    python eval_sap.py --depths 1 3 6 --sweep 0 5 10 15 20 30 40 60 80
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

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
IOU_THRS = np.round(np.arange(0.5, 1.0, 0.05), 2)


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------- 几何

def iou_matrix(a, b):
    """a:(N,4) b:(M,4) xyxy -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    ab = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return (inter / np.maximum(aa + ab - inter, 1e-9)).astype(np.float32)


def greedy_tp(iou, thr):
    """按分数序(行序即分数降序)贪心匹配, 返回 tp 布尔数组 (D,)"""
    D, G = iou.shape
    tp = np.zeros(D, bool)
    if D == 0 or G == 0:
        return tp
    best = iou.max(axis=1)
    cand = np.nonzero(best >= thr)[0]
    if len(cand) == 0:
        return tp
    taken = np.zeros(G, bool)
    for d in cand:
        row = iou[d]
        masked = np.where(taken, -1.0, row)
        j = int(masked.argmax())
        if masked[j] >= thr:
            taken[j] = True
            tp[d] = True
    return tp


def voc_ap(rec, prec):
    """101 点插值 AP (COCO 口径)"""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.searchsorted(mrec, np.linspace(0, 1, 101), side='left')
    return float(np.mean(mpre[np.clip(idx, 0, len(mpre) - 1)]))


# ---------------------------------------------------------------- 数据

def load_gt(ann_path):
    """file_name -> (box(N,4) float32, cls(N,)) ; 无标注的帧也给空数组"""
    with open(ann_path, encoding='utf-8') as f:
        coco = json.load(f)
    id2name = {im['id']: im['file_name'] for im in coco['images']}
    acc = {}
    for a in coco['annotations']:
        fn = id2name[a['image_id']]
        x, y, w, h = a['bbox']
        acc.setdefault(fn, []).append((x, y, x + w, y + h, int(a['category_id']) - 1))
    out = {}
    for fn in set(id2name.values()):
        v = acc.get(fn)
        if v:
            arr = np.asarray(v, np.float32)
            out[fn] = (arr[:, :4], arr[:, 4].astype(np.int64))
        else:
            out[fn] = (np.zeros((0, 4), np.float32), np.zeros(0, np.int64))
    return out


def build_timeline(stream_index):
    """按场景顺序展开帧; 返回 frames 列表与每场景时间戳(ms)"""
    with open(stream_index, encoding='utf-8') as f:
        idx = json.load(f)
    frames, scene_t, scene_files = [], {}, {}
    for s in idx['scenes']:
        ts, fns = [], []
        for fr in s['frames']:
            frames.append((s['scene_id'], fr['idx'], fr['t_ns'], fr['file']))
            ts.append(fr['t_ns'] / 1e6)
            fns.append(fr['file'])
        scene_t[s['scene_id']] = np.asarray(ts, np.float64)
        scene_files[s['scene_id']] = fns
    return frames, scene_t, scene_files


def assoc_floor(scene_t, scene_id, out_ms):
    """产出时刻 -> 不晚于它的最后一帧序号(floor); 越界则夹到场景边界"""
    ts = scene_t[scene_id]
    j = int(np.searchsorted(ts, out_ms, side='right')) - 1
    return 0 if j < 0 else (len(ts) - 1 if j >= len(ts) else j)


# ---------------------------------------------------------------- 调度

def schedule(compute_ms, t_ms, added_ms):
    """丢弃式实时调度; 返回被处理帧的索引与各自产出时刻(ms)"""
    n = len(t_ms)
    proc, out = [], []
    free = -1e18
    for i in range(n):
        if t_ms[i] < free:
            continue                     # 处理器忙, 这一帧被丢弃
        finish = t_ms[i] + compute_ms[i] + added_ms
        proc.append(i)
        out.append(finish)
        free = finish
    return np.asarray(proc, np.int64), np.asarray(out, np.float64)


# ---------------------------------------------------------------- 评测

def evaluate(rec, gt, frames, scene_t, scene_files, added_ms, verify=False,
             frame_key=None):
    box, score, cls, ptr = rec['box'], rec['score'], rec['cls'], rec['ptr']
    nrec = len(rec['scene'])
    # 录制文件可能只覆盖部分场景(试跑), 因此按 (scene, frame_idx) 映射回全局时序
    if frame_key is not None:
        gidx = np.fromiter(
            (frame_key.get((int(s), int(f)), -1)
             for s, f in zip(rec['scene'], rec['frame_idx'])),
            np.int64, nrec)
        if (gidx < 0).any():
            raise SystemExit('[错误] 录制的帧不在时序表内, 检查是否用了不同的 '
                             'stream_index')
    else:
        gidx = np.arange(nrec)
        if nrec != len(frames):
            raise SystemExit(f'[错误] 录制 {nrec} 帧 != 时序表 {len(frames)} 帧')

    t_ms = np.array([frames[g][2] / 1e6 for g in gidx], np.float64)
    compute = rec['compute_ms'].astype(np.float64)
    nf = len(gidx)

    if verify:
        proc = np.arange(nf)
        out_ms = t_ms.copy()
    else:
        proc, out_ms = schedule(compute, t_ms, added_ms)

    # 汇总每个被处理帧的预测, 并算出它关联到的 GT 帧
    boxes, scores, classes, assoc, counts = [], [], [], [], []
    for k, i in enumerate(proc):
        s = ptr[i]
        e = ptr[i + 1]
        if e <= s:
            continue
        g = gidx[i]
        s_id = frames[g][0]
        j = assoc_floor(scene_t, s_id, out_ms[k])
        boxes.append(box[s:e])
        scores.append(score[s:e])
        classes.append(cls[s:e])
        assoc.append(scene_files[s_id][j])
        counts.append(e - s)
    n_proc = len(proc)

    if not boxes:
        return dict(sap50=0.0, sap=0.0, n_proc=n_proc,
                    drop_pct=100.0 * (1 - n_proc / nf))

    box_a = np.concatenate(boxes)
    score_a = np.concatenate(scores)
    cls_a = np.concatenate(classes)
    # 关联帧要展开到【每个预测】一条, 否则无法随预测一起排序
    assoc_a = np.repeat(np.asarray(assoc), np.asarray(counts, np.int64))

    # 全局按分数降序 —— 这是 AP 的定义序
    order = np.argsort(-score_a, kind='stable')
    box_a, score_a, cls_a, assoc_a = (box_a[order], score_a[order],
                                      cls_a[order], assoc_a[order])

    # 编码 (关联帧, 类别) 为整数键, 便于快速分组
    fnames = sorted(set(assoc_a.tolist()))
    f2i = {f: i for i, f in enumerate(fnames)}
    fid = np.fromiter((f2i[f] for f in assoc_a.tolist()), np.int64, len(assoc_a))
    key = fid * 8 + cls_a

    # 分组: 组内保持分数降序
    gorder = np.argsort(key, kind='stable')
    sk = key[gorder]
    uniq, starts = np.unique(sk, return_index=True)
    ends = np.append(starts[1:], len(sk))

    # n_gt: 真实框总数必须覆盖【评测范围内所有帧】, 不能只覆盖"被关联到"的帧。
    # 原因: 丢弃的帧不产生预测, 也就不会被关联; 若只按关联帧计分母, 丢帧的 GT
    # 就不进分母 —— 等于"丢弃免费", 高延迟下指标虚高、掩盖真实漏检。
    # 真实系统里相机照常出帧, 模型跳过它就是漏检, 必须计入 recall。
    scenes_in_rec = set(int(s) for s in rec['scene'])
    n_gt = np.zeros(8, np.int64)
    for s_id in scenes_in_rec:
        if s_id not in scene_files:
            continue
        for fn in scene_files[s_id]:
            _, gc = gt[fn]
            if len(gc):
                n_gt += np.bincount(gc, minlength=8)

    # 类内按分数降序的位置索引 —— AP 必须【逐类分别积分再平均】,
    # 把所有类混在一起积一条 PR 曲线得到的是 class-agnostic AP, 与
    # COCO/ultralytics 的 mAP 口径不同(实测能差 3pp 以上)。
    cls_pos = [np.nonzero(cls_a == c)[0] for c in range(8)]

    aps = []
    for thr in IOU_THRS:
        tp = np.zeros(len(box_a), bool)
        for gi in range(len(uniq)):
            pos = gorder[starts[gi]:ends[gi]]        # 该组的全局位置(分数降序)
            u = uniq[gi]
            c = int(u % 8)
            f = fnames[int(u // 8)]
            gb, gc = gt[f]
            m = gc == c
            if not m.any():
                continue
            tp[pos] = greedy_tp(iou_matrix(box_a[pos], gb[m]), thr)

        # 逐类积分: 无 GT 的类跳过(COCO 对这类置 -1 并排除在均值外)
        per_cls = []
        for c in range(8):
            if n_gt[c] == 0:
                continue
            idx = cls_pos[c]
            if len(idx) == 0:
                per_cls.append(0.0)          # 有 GT 但无预测 -> AP 记 0
                continue
            csum = np.cumsum(tp[idx])
            rec = csum / n_gt[c]
            prec = csum / np.arange(1, len(idx) + 1)
            per_cls.append(voc_ap(rec, prec))
        aps.append(float(np.mean(per_cls)) if per_cls else 0.0)
    aps = np.asarray(aps)
    return dict(sap50=float(aps[0]), sap=float(aps.mean()), aps=aps.tolist(),
                n_proc=n_proc, drop_pct=100.0 * (1 - n_proc / nf))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds-dir', default='stream_preds')
    ap.add_argument('--depths', nargs='+', type=int, default=[1, 3, 6])
    ap.add_argument('--ann', default=os.path.join(DATA, 'annotations_val.json'))
    ap.add_argument('--stream-index', default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--sweep', nargs='+', type=float, default=None)
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--out', default='out_sap.json')
    args = ap.parse_args()

    frames, scene_t, scene_files = build_timeline(args.stream_index)
    log(f'时序: {len(frames)} 帧 / {len(scene_t)} 场景')
    frame_key = {(f[0], f[1]): i for i, f in enumerate(frames)}
    gt = load_gt(args.ann)
    log(f'GT: {sum(len(v[0]) for v in gt.values())} 框')

    sweep = args.sweep if args.sweep else [0.0]
    results = {}
    for d in args.depths:
        p = os.path.join(args.preds_dir, f'preds_d{d}.npz')
        if not os.path.exists(p):
            log(f'[跳过] 缺 {p}')
            continue
        z = np.load(p, allow_pickle=False)
        rec = {k: z[k] for k in z.files}
        log('')
        log('=' * 66)
        log(f'depth={d}  帧 {len(rec["scene"])}  推理耗时中位 '
            f'{np.median(rec["compute_ms"]):.2f}ms  P95 '
            f'{np.percentile(rec["compute_ms"], 95):.2f}ms')
        log('=' * 66)
        results[d] = []
        for L in sweep:
            t0 = time.time()
            r = evaluate(rec, gt, frames, scene_t, scene_files, L,
                         verify=args.verify, frame_key=frame_key)
            r['added_ms'] = L
            results[d].append(r)
            log(f'  L={L:6.1f}ms  sAP50={r["sap50"]:.4f}  sAP={r["sap"]:.4f}  '
                f'处理 {r["n_proc"]:5d}  丢弃 {r["drop_pct"]:5.1f}%  '
                f'({time.time()-t0:.0f}s)')

    log('')
    log('=' * 66)
    if args.verify:
        REF = {1: 0.3363, 3: 0.3865, 6: 0.4053}
        log('自检: AP50 应等于 ultralytics 的 mAP50')
        log(f"{'深度':<6}{'本脚本 AP50':>14}{'ultralytics':>14}{'差值':>10}")
        for d in args.depths:
            if d in results:
                got = results[d][0]['sap50']
                ref = REF.get(d, float('nan'))
                log(f'{d:<6}{got:>14.4f}{ref:>14.4f}{got-ref:>+10.4f}')
    else:
        hdr = f"{'注入延迟(ms)':>12}"
        for d in args.depths:
            hdr += f'{str(d)+"层 sAP50":>13}{"丢弃%":>8}'
        log(hdr)
        for i, L in enumerate(sweep):
            row = f'{L:>12.0f}'
            for d in args.depths:
                if d in results:
                    row += f'{results[d][i]["sap50"]:>13.4f}{results[d][i]["drop_pct"]:>8.1f}'
            log(row)
    log('=' * 66)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(dict(verify=args.verify, sweep=sweep, results=results),
                  f, ensure_ascii=False, indent=2)
    log(f'已写入 {args.out}')


if __name__ == '__main__':
    main()
