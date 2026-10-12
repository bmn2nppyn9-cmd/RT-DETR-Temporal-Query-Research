# -*- coding: utf-8 -*-
"""
实验 A+B:自适应解码创新点的可行性判定
========================================
回答一个问题:按帧自适应选择 Decoder 层数,是否有价值?

设计思路(为什么不只是"跑 1/3/6 层看看精度差多少"):
  层数越多精度越高是常识, 单看这个证明不了创新点。
  真正要证明的是「不同帧的最优层数不同, 且这个差异可被内容预测」。
  因此本脚本做两件事:
    实验A  固定层数基线曲线  -> 建立"省层数的代价"基线
    实验B  Oracle 增益判定   -> 上帝视角下, 按帧自适应最多能赚多少

  判据(实验B):
    Oracle增益 < 1.0 sAP  -> 创新点不成立, 建议放弃路由设计
    1.0 ~ 2.0 sAP         -> 空间有限, 降级为次要贡献
    > 2.0 sAP             -> 成立, 继续做真实路由器对照

两段式设计:
  1) infer  阶段: 跑推理, 把「每帧 x 每层数」的逐帧评分矩阵存盘 (耗时, 只跑一次)
  2) analyze 阶段: 读盘做 Oracle 统计与判定 (秒级, 可反复重跑)

用法:
  # 0) 先自检(无需数据/权重, 用合成数据验证流程)
  python exp_ab_adaptive_depth.py smoke

  # 1) 真实数据推理(需要数据集 + 训练好的权重)
  python exp_ab_adaptive_depth.py infer \
      --images D:/Argoverse-HD/val/images \
      --ann    D:/Argoverse-HD/val/annotations.json \
      --weights rtdetr-l-argoverse.pt \
      --depths 1 2 3 4 5 6 \
      --out out_ab

  # 2) 分析与判定(读上一步产物, 秒级)
  python exp_ab_adaptive_depth.py analyze --out out_ab

  # 3) 画图
  python exp_ab_adaptive_depth.py plot --out out_ab

依赖: torch, ultralytics, numpy, PIL, (可选 scipy/pandas/matplotlib)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

# Windows 控制台默认 GBK, 无法输出 ✓/⚠ 等符号, 统一切到 UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ---------------------------------------------------------------- 工具


def log(msg):
    print(msg, flush=True)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) b: (M,4), 格式 xyxy"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    # 广播求交
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def class_aware_iou(dets_xyxy, dets_cls, gts_xyxy, gts_cls):
    """类别感知的 IoU 矩阵: 类别不一致的位置置 0, 从而无法匹配。

    这是 COCO 评测的标准做法 —— "把行人检成汽车"不应算正确。
    """
    iou = iou_matrix(dets_xyxy, gts_xyxy)
    if iou.size == 0:
        return iou
    same = (dets_cls[:, None] == gts_cls[None, :])
    return np.where(same, iou, 0.0).astype(np.float32)


def greedy_match(iou: np.ndarray, thr: float):
    """按 IoU 从高到低贪心一对一匹配, 返回 (匹配对列表, 未匹配det索引, 未匹配gt索引)"""
    if iou.size == 0:
        return [], list(range(iou.shape[0])), list(range(iou.shape[1]))
    pairs = []
    used_d = set()
    used_g = set()
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for d, g in order:
        if iou[d, g] < thr:
            break
        if d in used_d or g in used_g:
            continue
        used_d.add(int(d)); used_g.add(int(g))
        pairs.append((int(d), int(g)))
    return pairs, [i for i in range(iou.shape[0]) if i not in used_d], \
                  [j for j in range(iou.shape[1]) if j not in used_g]


def per_frame_ap(dets: np.ndarray, gts: np.ndarray, iou_thrs=None) -> float:
    """单帧 AP(COCO 式 101 点插值), 无检测/无真值时的约定处理。

    dets: (N,5) [x1,y1,x2,y2,score]  gts: (M,4)
    """
    if iou_thrs is None:
        iou_thrs = np.arange(0.5, 1.0, 0.05)
    if len(gts) == 0 and len(dets) == 0:
        return float('nan')          # 双方皆空, 不计入
    if len(gts) == 0:
        return 0.0                   # 误检, 得 0 分
    if len(dets) == 0:
        return 0.0                   # 漏检, 得 0 分

    aps = []
    for thr in iou_thrs:
        iou = iou_matrix(dets[:, :4], gts)
        _, _, _ = greedy_match(iou, thr)
        # 按分数降序的 TP/FP 序列
        order = np.argsort(-dets[:, 4])
        matched_g = set()
        tp = np.zeros(len(dets), dtype=np.float32)
        fp = np.zeros(len(dets), dtype=np.float32)
        for rank, di in enumerate(order):
            row = iou[di]
            # 在未占用的真值中取最大 IoU
            best_g, best_v = -1, 0.0
            for gj in range(len(gts)):
                if gj in matched_g:
                    continue
                if row[gj] > best_v:
                    best_v, best_g = row[gj], gj
            if best_g >= 0 and best_v >= thr:
                matched_g.add(best_g); tp[rank] = 1
            else:
                fp[rank] = 1
        tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
        rec = tp_cum / len(gts)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        # 101 点插值
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.searchsorted(mrec, np.linspace(0, 1, 101), side='left')
        idx = np.clip(idx, 0, len(mpre) - 1)
        aps.append(float(np.mean(mpre[idx])))
    return float(np.mean(aps))


def per_frame_f1(dets: np.ndarray, gts: np.ndarray, thr: float = 0.5) -> float:
    """单帧 F1@IoU, 作为 AP 的稳健补充指标"""
    if len(gts) == 0 and len(dets) == 0:
        return float('nan')
    if len(gts) == 0 or len(dets) == 0:
        return 0.0
    iou = iou_matrix(dets[:, :4], gts)
    pairs, _, _ = greedy_match(iou, thr)
    tp = len(pairs)
    fp = len(dets) - tp
    fn = len(gts) - tp
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def motion_iou(prev_gts: np.ndarray, cur_gts: np.ndarray) -> float:
    """帧间真值运动 IoU 均值(FGFA 的 motion IoU 定义), 越小说明变化越剧烈"""
    if len(prev_gts) == 0 or len(cur_gts) == 0:
        return float('nan')
    iou = iou_matrix(cur_gts, prev_gts)
    return float(iou.max(axis=1).mean())


# ---------------------------------------------------------------- 解码层数控制


class DepthController:
    """临时修改 RT-DETR 解码层数。

    依据: ultralytics 的 DeformableTransformerDecoder 推理时靠
    `i == self.eval_idx` 触发 break, 因此只需改 eval_idx 即可截断,
    无需真的切 layers(实测已验证)。
    """

    def __init__(self, model):
        self.dec = None
        # 找到 RTDETRDecoder
        for m in model.modules():
            if type(m).__name__ == 'RTDETRDecoder' and hasattr(m, 'decoder'):
                self.dec = m
                break
        if self.dec is None:
            raise RuntimeError('未找到 RTDETRDecoder, 请确认权重是 RT-DETR')
        self.full = self.dec.decoder.eval_idx + 1
        self._saved = None

    def set(self, n):
        n = min(max(1, n), self.full)
        self._saved = self.dec.decoder.eval_idx
        self.dec.decoder.eval_idx = n - 1

    def restore(self):
        if self._saved is not None:
            self.dec.decoder.eval_idx = self._saved


# ---------------------------------------------------------------- 推理阶段


def load_gt(ann_path):
    """读取 COCO 格式标注 -> {image_id: (M,4) xyxy}, 以及 image_id -> file_name"""
    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    img_info = {im['id']: im for im in coco['images']}
    gt = defaultdict(list)
    for a in coco['annotations']:
        if a.get('iscrowd', 0):
            continue
        x, y, w, h = a['bbox']
        gt[a['image_id']].append([x, y, x + w, y + h])
    gt = {k: np.array(v, dtype=np.float32) for k, v in gt.items()}
    return img_info, gt


def run_infer(args):
    import torch
    from PIL import Image
    from ultralytics import RTDETR

    os.makedirs(args.out, exist_ok=True)
    device = args.device

    log(f'[1/4] 加载权重: {args.weights}')
    model = RTDETR(args.weights)
    model.model.to(device).eval()
    ctrl = DepthController(model.model)
    log(f'      模型完整解码层数 = {ctrl.full}')

    log(f'[2/4] 读取标注: {args.ann}')
    img_info, gt = load_gt(args.ann)
    img_ids = sorted(img_info.keys())
    if args.limit:
        img_ids = img_ids[:args.limit]
    log(f'      图像 {len(img_ids)} 张, 待测层数 {args.depths}')

    # 预热: 避免首次 CUDA 调用(preprocessing/算子编译)污染首帧耗时
    log('      预热中...')
    try:
        from PIL import Image as _I
        _p = os.path.join(args.images, img_info[img_ids[0]]['file_name'])
        if os.path.exists(_p):
            _im = _I.open(_p).convert('RGB')
            for _d in (args.depths[0], args.depths[-1]):
                ctrl.set(_d)
                try:
                    model.predict(_im, conf=args.conf, iou=args.iou,
                                  verbose=False, device=device)
                finally:
                    ctrl.restore()
            if device == 'cuda':
                torch.cuda.synchronize()
            log('      预热完成')
    except Exception as e:
        log(f'      预热跳过: {e}')

    # 输出结构: rows = [ {frame, depth, ap, f1, n_det, n_gt, motion_iou, t_ms} ]
    rows = []
    log('[3/4] 逐帧逐层数推理...')
    t_start = time.time()

    for fi, iid in enumerate(img_ids):
        info = img_info[iid]
        path = os.path.join(args.images, info['file_name'])
        if not os.path.exists(path):
            log(f'      [警告] 图片缺失, 跳过: {path}')
            continue
        img = Image.open(path).convert('RGB')
        g = gt.get(iid, np.zeros((0, 4), dtype=np.float32))

        # 帧间运动 IoU(与前一帧的真值比)
        if fi > 0:
            prev = gt.get(img_ids[fi - 1], np.zeros((0, 4), dtype=np.float32))
            miou = motion_iou(prev, g)
        else:
            miou = float('nan')

        for d in args.depths:
            ctrl.set(d)
            try:
                with torch.no_grad():
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    res = model.predict(img, conf=args.conf, iou=args.iou,
                                        verbose=False, device=device)[0]
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    t_ms = (time.perf_counter() - t0) * 1000
                if res.boxes is not None and len(res.boxes) > 0:
                    b = res.boxes.xyxy.cpu().numpy()
                    s = res.boxes.conf.cpu().numpy()[:, None]
                    dets = np.concatenate([b, s], axis=1)
                else:
                    dets = np.zeros((0, 5), dtype=np.float32)
            except Exception as e:
                log(f'      [错误] frame={iid} depth={d}: {type(e).__name__}: {e}')
                continue
            finally:
                ctrl.restore()

            rows.append(dict(
                frame=iid, depth=d,
                ap=per_frame_ap(dets, g),
                f1=per_frame_f1(dets, g),
                n_det=int(len(dets)), n_gt=int(len(g)),
                motion_iou=miou, t_ms=t_ms,
            ))

        if (fi + 1) % args.log_every == 0:
            el = time.time() - t_start
            log(f'      进度 {fi+1}/{len(img_ids)}  用时 {el/60:.1f} 分钟')

    out_csv = os.path.join(args.out, 'per_frame_scores.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f'[4/4] 逐帧结果已存: {out_csv}  ({len(rows)} 行)')
    log('      下一步: python exp_ab_adaptive_depth.py analyze --out ' + args.out)


# ---------------------------------------------------------------- 分析阶段


def load_rows(out_dir):
    path = os.path.join(out_dir, 'per_frame_scores.csv')
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            r['depth'] = int(r['depth'])
            for k in ('ap', 'f1', 'motion_iou', 't_ms'):
                r[k] = float(r[k])
            for k in ('n_det', 'n_gt'):
                r[k] = int(r[k])
            r['frame'] = int(r['frame'])
            rows.append(r)
    return rows


def run_analyze(args):
    if not hasattr(args, 'eps'):
        args.eps = 0.005
    rows = load_rows(args.out)
    depths = sorted({r['depth'] for r in rows})
    frames = sorted({r['frame'] for r in rows})

    # 构建矩阵 frame x depth (AP 与 F1)
    def build(key):
        M = {}
        for r in rows:
            v = r[key]
            if not np.isnan(v):
                M[(r['frame'], r['depth'])] = v
        return M
    AP, F1 = build('ap'), build('f1')

    # 只保留所有深度都有值的帧(公平比较)
    valid = [fr for fr in frames if all((fr, d) in AP for d in depths)]
    if not valid:
        log('[!] 没有有效帧, 请检查推理阶段是否正常完成')
        return
    log(f'有效帧数: {len(valid)} / {len(frames)}   层数档位: {depths}\n')

    def mat(M):
        return np.array([[M[(fr, d)] for d in depths] for fr in valid], dtype=np.float64)

    ap_mat, f1_mat = mat(AP), mat(F1)
    ap_mean = np.nanmean(ap_mat, axis=1)
    if not np.all(np.isfinite(ap_mean)) or ap_mean.std() == 0:
        log('[!] AP 全为常数或无效, 无法分析(常见原因: 权重未训练/数据不匹配)')
        return

    # ---------- 实验A: 固定层数曲线 ----------
    log('=' * 78)
    log('  实验A: 固定 Decoder 层数下的精度(全帧统一用 N 层)')
    log('=' * 78)
    log(f'  {"层数":>4} | {"AP均值":>8} | {"相对满层":>9} | {"F1均值":>8} | {"帧均耗时":>9}')
    log('  ' + '-' * 62)
    # 耗时用中位数(抗首帧预热/系统抖动等异常值)
    tmap = {d: float(np.median([r['t_ms'] for r in rows if r['depth'] == d])) for d in depths}
    fixed_ap = {d: float(np.mean(ap_mat[:, j])) for j, d in enumerate(depths)}
    ap_max_depth = fixed_ap[depths[-1]]
    for j, d in enumerate(depths):
        m = fixed_ap[d]
        delta = (m - ap_max_depth) * 100 if d != depths[-1] else 0.0
        f1m = float(np.mean(f1_mat[:, j]))
        log(f'  {d:>4} | {m:8.4f} | {delta:>+8.2f}% | {f1m:8.4f} | {tmap[d]:7.1f}ms')
    best_fixed_d = max(fixed_ap, key=fixed_ap.get)
    log(f'\n  最优固定层数 = {best_fixed_d} 层 (AP={fixed_ap[best_fixed_d]:.4f})')

    # ---------- 实验B: Oracle 增益 ----------
    log('\n' + '=' * 78)
    log('  实验B: Oracle 增益(上帝视角按帧选最优层数)')
    log('=' * 78)
    oracle_ap = float(np.mean(ap_mat.max(axis=1)))
    gain_ap = (oracle_ap - fixed_ap[best_fixed_d]) * 100
    oracle_f1 = float(np.nanmean(f1_mat.max(axis=1)))
    best_fixed_f1 = max(float(np.mean(f1_mat[:, j])) for j in range(len(depths)))
    gain_f1 = (oracle_f1 - best_fixed_f1) * 100

    log(f'  最优固定层数 AP      : {fixed_ap[best_fixed_d]:.4f}')
    log(f'  Oracle AP(逐帧最优)  : {oracle_ap:.4f}')
    log(f'  >>> Oracle 增益(AP)  : {gain_ap:+.2f} 个百分点')
    log(f'  >>> Oracle 增益(F1)  : {gain_f1:+.2f} 个百分点')

    # 各层数被选为最优的频率(说明"层数需求是否真的因帧而异")
    best_idx = ap_mat.argmax(axis=1)
    freq = {d: float(np.mean(best_idx == j)) * 100 for j, d in enumerate(depths)}
    log('\n  各帧最优层数的分布(说明需求是否因帧而异):')
    for d in depths:
        bar = '#' * int(freq[d] / 2.5)
        log(f'    {d:>3} 层: {freq[d]:5.1f}%  {bar}')
    entropy = -sum(p / 100 * np.log(p / 100 + 1e-12) for p in freq.values())
    log(f'    分布熵 = {entropy:.3f}  (越高说明需求越分散, 自适应空间越大)')

    # ---------- 实验B-2: 成本侧 Oracle(关键!) ----------
    # 只看精度增益会漏掉自适应的另一半价值: 易帧少算层数、精度不损失。
    # 因此这里问: 保持精度不变的前提下, 平均能用几层?
    log('\n' + '=' * 78)
    log('  实验B-2: 成本侧 Oracle —— 保持精度, 平均能省几层?')
    log('=' * 78)
    lat_vec = np.array([tmap[d] for d in depths], dtype=np.float64)
    peak = ap_mat.max(axis=1)
    eps = args.eps
    min_idx = np.array([
        next(j for j in range(len(depths)) if ap_mat[i, j] >= peak[i] - eps)
        for i in range(len(valid))
    ])
    min_depths = np.array([depths[j] for j in min_idx], dtype=float)
    acc_msd = float(np.mean([ap_mat[i, min_idx[i]] for i in range(len(valid))]))
    lat_msd = float(np.mean([lat_vec[min_idx[i]] for i in range(len(valid))]))
    lat_full = float(lat_vec[-1])
    acc_full = fixed_ap[depths[-1]]
    save_pct = (1 - lat_msd / lat_full) * 100

    log(f'  容差 eps = {eps*100:.2f} 个百分点(低于峰值此幅度即认为"够用")')
    log(f'  最小充分层数策略 : 平均 {min_depths.mean():.2f} 层,  帧均 {lat_msd:6.2f} ms,  AP {acc_msd:.4f}')
    log(f'  固定满层({depths[-1]} 层)   : 平均 {depths[-1]:.2f} 层,  帧均 {lat_full:6.2f} ms,  AP {acc_full:.4f}')
    log(f'  >>> 等精度下节省耗时 : {save_pct:+.1f}%   (精度差 {(acc_msd-acc_full)*100:+.2f} pp)')
    mfreq = {d: float(np.mean(min_depths == d)) * 100 for d in depths}
    log('\n  最小充分层数的分布:')
    for d in depths:
        bar = '#' * int(mfreq[d] / 2.5)
        log(f'    {d:>3} 层: {mfreq[d]:5.1f}%  {bar}')

    # ---------- 实验B-3: Oracle 帕累托前沿 ----------
    log('\n' + '=' * 78)
    log('  实验B-3: Oracle 帕累托前沿(精度 vs 耗时)')
    log('=' * 78)
    pts = []
    for lam in np.linspace(0.0, 80.0, 161):
        util = ap_mat - lam * lat_vec[None, :] / 1000.0
        j = util.argmax(axis=1)
        a = float(np.mean([ap_mat[i, j[i]] for i in range(len(valid))]))
        l = float(np.mean([lat_vec[j[i]] for i in range(len(valid))]))
        pts.append((l, a))
    pts.sort()
    front, best_a = [], -1.0
    for l, a in pts:
        if a > best_a + 1e-9:
            front.append((l, a)); best_a = a
    log(f'  {"帧均耗时":>10} | {"AP":>8} | {"等效平均层数":>12}')
    log('  ' + '-' * 36)
    step = max(1, len(front) // 8)          # 只抽样显示, 避免刷屏
    for l, a in front[::step] + [front[-1]]:
        nd = np.interp(l, lat_vec, depths)
        log(f'  {l:8.2f}ms | {a:8.4f} | {nd:11.2f}')
    lat_at_full_acc = min((l for l, a in front if a >= acc_full - 1e-9), default=lat_full)
    gain_lat_pareto = (1 - lat_at_full_acc / lat_full) * 100
    log(f'\n  >>> 达到满层精度所需最小耗时: {lat_at_full_acc:.2f}ms '
        f'(满层 {lat_full:.2f}ms, 省 {gain_lat_pareto:.1f}%)')

    # 严格对照: 最优"固定"层数策略在等精度下要多少耗时?
    # 若固定策略也能便宜地达到该精度, 则逐帧自适应没有意义。
    fixed_candidates = [(lat_vec[j], fixed_ap[d], d)
                        for j, d in enumerate(depths)
                        if fixed_ap[d] >= acc_full - 1e-9]
    if fixed_candidates:
        lat_best_fixed_eq = min(c[0] for c in fixed_candidates)
        gain_over_best_fixed = (1 - lat_at_full_acc / lat_best_fixed_eq) * 100
        d_used = [c[2] for c in fixed_candidates if c[0] == lat_best_fixed_eq][0]
        log(f'  >>> 对照: 最优固定策略({d_used} 层, {lat_best_fixed_eq:.2f}ms)等精度下需 '
            f'{lat_best_fixed_eq:.2f}ms, 逐帧自适应再省 {gain_over_best_fixed:+.1f}%')
        log('      (后者才是"自适应路由"相对"静态选择"的真实上界增益)')
    else:
        gain_over_best_fixed = gain_lat_pareto
        log('  (所有固定策略都达不到满层精度, 逐帧自适应的省时即全部来自自适应)')

    # ---------- 可预测性: 帧间变化 vs 最优层数 ----------
    log('\n' + '=' * 78)
    log('  可预测性检验: 帧间变化强度能否解释"该用几层"?')
    log('=' * 78)
    miou = np.array([next(r['motion_iou'] for r in rows
                          if r['frame'] == fr and r['depth'] == depths[0])
                     for fr in valid], dtype=np.float64)
    opt_d = np.array([depths[i] for i in best_idx], dtype=np.float64)
    mask = np.isfinite(miou)
    if mask.sum() > 10:
        try:
            from scipy.stats import spearmanr, pearsonr
            rho, p_sp = spearmanr(miou[mask], opt_d[mask])
            r_p, p_pe = pearsonr(miou[mask], opt_d[mask])
            log(f'  Spearman(motion_IoU, 最优层数) = {rho:+.3f}  (p={p_sp:.2e})')
            log(f'  Pearson (motion_IoU, 最优层数) = {r_p:+.3f}  (p={p_pe:.2e})')
            log('  解读: 期望为「负相关」——变化越剧烈(motion_IoU 越低)越需要深层。')
            if abs(rho) < 0.1:
                log('        ⚠ 相关性很弱: motion_IoU 不足以预测最优层数, 路由器可能学不动。')
            else:
                log('        ✓ 存在可用的相关性, 路由器有可学的信号。')
        except ImportError:
            log('  (未安装 scipy, 跳过相关性检验)')
    else:
        log('  motion_IoU 有效样本不足, 跳过(单帧或真值缺失?)。')

    # ---------- 判定(同时看精度增益与成本节省) ----------
    log('\n' + '=' * 78)
    log('  【判定】')
    log('=' * 78)
    log(f'  精度侧 Oracle 增益 : {gain_ap:+.2f} 个百分点')
    log(f'  成本侧等精度节省   : {save_pct:+.1f}%')
    log(f'  逐帧自适应 vs 最优固定策略的省时上界: {gain_over_best_fixed:+.1f}%')

    if gain_ap >= 2.0:
        verdict = '成立(精度侧)'
        advice = ('精度侧 Oracle 增益 > 2.0 个百分点: 按帧自适应能明显提精度。\n'
                  '  → 建议: 全力推进。继续做实验C(真实路由器 vs 随机路由 vs Oracle 三方对照),\n'
                  '     并重点论证路由信号的可学性(见下方相关性检验)。')
    elif gain_over_best_fixed >= 18.0:
        verdict = '成立(成本侧)'
        advice = ('精度增益有限(<2.0pp), 但等精度下逐帧自适应比最优固定策略省 >18% 耗时 ——\n'
                  '  这是自适应路由的真实上界价值所在。\n'
                  '  → 建议: 把论文的论述重点从"提精度"改为"等精度降本"。\n'
                  '     端侧场景最关心单位算力的精度, 这个论点无需重训检测器即可受益。\n'
                  '     但仍须做实验C证明"学到的路由器能接近这个上界"。')
    elif gain_ap >= 1.0 or gain_over_best_fixed >= 10.0:
        verdict = '空间有限'
        advice = ('两方向增益都不大。\n'
                  '  → 建议: 降级为次要贡献。主贡献放在查询传播 + 端侧结构分析上,\n'
                  '     路由作为"锦上添花"的消融项, 并诚实报告增益幅度。')
    else:
        verdict = '不成立'
        advice = ('精度与成本两侧收益都很小: 即使上帝视角也赚不到什么。\n'
                  '  → 建议: 放弃"按帧路由层数"作为主创新点。\n'
                  '     可改为「固定轻量 Decoder + 查询传播」的工程优化路线, 或转向其它维度\n'
                  '     (如按帧选择分辨率/关键帧率, 在实验中往往增益更大)。')
    log(f'\n  结论: 创新点【{verdict}】')
    log('  ' + advice.replace('\n', '\n  '))

    result = dict(
        n_frames=len(valid), depths=depths,
        fixed_ap={str(d): fixed_ap[d] for d in depths},
        best_fixed_depth=int(best_fixed_d), best_fixed_ap=fixed_ap[best_fixed_d],
        oracle_ap=oracle_ap, oracle_gain_pp=gain_ap,
        oracle_gain_f1_pp=gain_f1,
        best_depth_freq={str(d): freq[d] for d in depths},
        depth_entropy=float(entropy),
        # 成本侧
        mean_sufficient_depth=float(min_depths.mean()),
        sufficient_depth_freq={str(d): mfreq[d] for d in depths},
        latency_msd_ms=lat_msd, latency_full_ms=lat_full,
        equal_acc_saving_pct=save_pct,
        pareto_saving_pct=gain_lat_pareto,
        adaptive_vs_best_fixed_saving_pct=float(gain_over_best_fixed),
        mean_latency_ms={str(d): tmap[d] for d in depths},
        verdict=verdict,
    )
    with open(os.path.join(args.out, 'oracle_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f'\n  结果已存: {os.path.join(args.out, "oracle_analysis.json")}')

    # 保存矩阵供画图
    np.savez(os.path.join(args.out, 'matrices.npz'),
             ap=ap_mat, f1=f1_mat, depths=np.array(depths),
             frames=np.array(valid), motion=miou)
    return result


# ---------------------------------------------------------------- 画图


def run_plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.dpi'] = 300

    z = np.load(os.path.join(args.out, 'matrices.npz'))
    ap, depths, miou = z['ap'], z['depths'], z['motion']
    with open(os.path.join(args.out, 'oracle_analysis.json'), encoding='utf-8') as f:
        res = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

    # (a) 固定层数曲线 vs Oracle
    ax = axes[0]
    m = ap.mean(axis=0)
    ax.plot(depths, m, 'o-', color='#4C72B0', lw=2.4, ms=8, label='固定层数')
    ax.axhline(res['oracle_ap'], color='#C44E52', ls='--', lw=2.2,
               label=f"Oracle = {res['oracle_ap']:.4f}")
    bf = res['best_fixed_depth']
    ax.scatter([bf], [res['best_fixed_ap']], s=180, facecolor='none',
               edgecolor='#55A868', lw=2.5, zorder=5, label=f'最优固定({bf}层)')
    ax.annotate(f"+{res['oracle_gain_pp']:.2f}pp", xy=(depths[-1], res['oracle_ap']),
                xytext=(-8, -26), textcoords='offset points', color='#C44E52',
                fontweight='bold', fontsize=11)
    ax.set_xlabel('Decoder 层数'); ax.set_ylabel('AP')
    ax.set_title('(a) 固定层数曲线 与 Oracle 上界', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, ls=':')

    # (b) 最优层数分布
    ax = axes[1]
    freq = [res['best_depth_freq'][str(int(d))] for d in depths]
    bars = ax.bar([str(int(d)) for d in depths], freq, color='#55A868', edgecolor='white')
    for b, v in zip(bars, freq):
        ax.text(b.get_x() + b.get_width()/2, v + max(freq)*0.02, f'{v:.0f}%',
                ha='center', fontsize=9.5, fontweight='bold')
    ax.set_xlabel('最优 Decoder 层数'); ax.set_ylabel('占比 (%)')
    ax.set_title(f'(b) 各帧最优层数分布（熵={res["depth_entropy"]:.2f}）', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, ls=':'); ax.set_axisbelow(True)

    # (c) 可预测性散点
    ax = axes[2]
    opt = np.array([depths[i] for i in ap.argmax(axis=1)])
    msk = np.isfinite(miou)
    ax.scatter(miou[msk], opt[msk] + np.random.uniform(-0.12, 0.12, msk.sum()),
               s=9, alpha=0.35, color='#4C72B0')
    ax.set_xlabel('帧间运动 IoU（越低=变化越剧烈）')
    ax.set_ylabel('该帧最优层数')
    ax.set_title('(c) 可预测性：变化强度 vs 最优层数', fontweight='bold')
    ax.grid(alpha=0.3, ls=':')

    fig.suptitle(f'实验A+B 结果   判定：{res["verdict"]}   '
                 f'Oracle 增益 {res["oracle_gain_pp"]:+.2f} 个百分点',
                 fontsize=13, fontweight='bold', y=1.03)
    p = os.path.join(args.out, 'ab_analysis.png')
    plt.close(fig)
    fig.savefig(p)
    print('图已保存:', p)


# ---------------------------------------------------------------- 自检模式


def run_smoke(args):
    """无需数据集与权重: 用合成数据验证分析链路是否正常"""
    log('=' * 78)
    log('  自检模式: 合成数据验证"分析+判定"链路')
    log('=' * 78)
    rng = np.random.default_rng(0)
    out = os.path.join(args.out)
    os.makedirs(out, exist_ok=True)

    depths = args.depths
    n = 400
    rows = []
    # 构造"两面都有"的场景:
    #   难帧(变化剧烈)  : 层数越多精度越高, 需满层  -> 精度侧收益
    #   易帧(变化平缓)  : 浅层即达上限, 再深没有增益 -> 成本侧收益(可早停)
    for fr in range(n):
        hard = rng.random() < 0.45
        if hard:
            miou = float(rng.uniform(0.25, 0.55))
            # 峰值 0.40, 需 6 层才够
            curve = 0.20 + 0.20 * ((np.arange(len(depths))) / (len(depths) - 1)) ** 0.7
        else:
            miou = float(rng.uniform(0.85, 0.99))
            # 峰值 0.40, 2 层即达, 之后平坦
            curve = 0.30 + 0.10 * np.minimum(np.arange(len(depths)) / 1.0, 1.0)
        for j, d in enumerate(depths):
            ap = float(curve[j] + rng.uniform(-0.008, 0.008))
            rows.append(dict(frame=fr, depth=d, ap=ap, f1=ap,
                             n_det=int(rng.integers(5, 40)), n_gt=int(rng.integers(5, 40)),
                             motion_iou=miou, t_ms=float(2.8 + (d - 1) * 1.3)))
    with open(os.path.join(out, 'per_frame_scores.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log(f'  已生成合成数据: {os.path.join(out, "per_frame_scores.csv")} ({len(rows)} 行)')
    log('  合成数据构成: 45% 困难帧(需满层) + 55% 平缓帧(浅层即够)。')
    log('  预期: 精度侧增益中等, 成本侧应有明显节省 —— 用于验证两条分析链路都通。\n')

    class A:  # 复用 analyze
        pass
    a = A(); a.out = out
    run_analyze(a)
    try:
        b = A(); b.out = out
        run_plot(b)
    except Exception as e:
        log(f'  (画图跳过: {e})')


# ---------------------------------------------------------------- 入口


def main():
    ap = argparse.ArgumentParser(
        description='实验A+B: 自适应解码创新点可行性判定',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('infer', help='推理: 逐帧逐层数评测并存盘')
    p1.add_argument('--images', required=True, help='图片目录')
    p1.add_argument('--ann', required=True, help='COCO 格式标注 json')
    p1.add_argument('--weights', required=True, help='训练好的 RT-DETR 权重 (.pt)')
    p1.add_argument('--depths', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6])
    p1.add_argument('--out', default='out_ab')
    p1.add_argument('--device', default='cuda')
    p1.add_argument('--conf', type=float, default=0.001, help='低阈值以公平计分')
    p1.add_argument('--iou', type=float, default=0.7)
    p1.add_argument('--limit', type=int, default=0, help='只跑前 N 张(调试用)')
    p1.add_argument('--log-every', type=int, default=50)

    p2 = sub.add_parser('analyze', help='分析: Oracle 增益与判定')
    p2.add_argument('--out', default='out_ab')
    p2.add_argument('--eps', type=float, default=0.005,
                    help='最小充分层数的精度容差(默认 0.005 = 0.5 个百分点)')

    p3 = sub.add_parser('plot', help='画图')
    p3.add_argument('--out', default='out_ab')

    p4 = sub.add_parser('smoke', help='自检: 用合成数据验证流程')
    p4.add_argument('--depths', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6])
    p4.add_argument('--out', default='out_ab_smoke')
    p4.add_argument('--eps', type=float, default=0.005)

    args = ap.parse_args()
    if args.cmd == 'infer':
        run_infer(args)
    elif args.cmd == 'analyze':
        run_analyze(args)
    elif args.cmd == 'plot':
        run_plot(args)
    elif args.cmd == 'smoke':
        run_smoke(args)


if __name__ == '__main__':
    main()
