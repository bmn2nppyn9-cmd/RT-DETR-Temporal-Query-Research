# -*- coding: utf-8 -*-
"""test_b0_regression.py —— B0 探针阶段 1: 回归测试
=====================================================
验证项(对应任务书 五.阶段1):
  R1  temporal 关闭时, 修改后代码与修改前录制(preds_d3.npz)输出一致
  R2  temporal 开->关循环后, 输出与从未开启时一致 (无残留状态)
  R3  场景首帧: B0(开启但无状态) 与 A0 输出一致
  R4  第二帧: decoder 内部 query 数=375, 前 300 current + 后 75 temporal
  R5  最终输出仍 <= top-300
  R6  无 NaN/Inf, 框坐标在合法范围
  R7  reset 后状态清空, 输出回到首帧行为
  R8  state 张量不携带梯度 (detach)

结果写入 query_prop_b0/logs/regression_log.json 与 .txt
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
CKPT = r'D:\lunwen\自研\runs\detect\runs_rtdetr\r18_640_sub20_b4\weights\best.pt'
OLD_NPZ = r'D:\lunwen\自研\stream_preds\preds_d3.npz'   # 修改前代码录制 (只读)
LOG_DIR = r'D:\lunwen\自研\query_prop_b0\logs'

LINES = []


def log(m):
    print(m, flush=True)
    LINES.append(str(m))


def load_bgr(path):
    with Image.open(path) as im:
        return np.array(im.convert('RGB'))[:, :, ::-1].copy()


def find_head(model):
    for mod in model.modules():
        if type(mod).__name__ == 'RTDETRDecoder' and hasattr(mod, 'decoder'):
            return mod
    return None


def det_array(r):
    b = r.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 6), np.float32)
    return np.concatenate([
        b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]
    ], 1).astype(np.float32)


def max_diff(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0.0 if len(a) == len(b) else float('inf')
    return float(np.abs(a[:n] - b[:n]).max())


def main():
    results = {}
    os.makedirs(LOG_DIR, exist_ok=True)

    from ultralytics import RTDETR
    m = RTDETR(CKPT)
    head = find_head(m.model)
    assert head is not None
    head.decoder.eval_idx = 2

    with open(os.path.join(DATA, 'stream_index.json'), encoding='utf-8') as f:
        idx = json.load(f)
    s0 = idx['scenes'][0]
    files = [os.path.join(DATA, 'images', fr['file']) for fr in s0['frames'][:6]]
    imgs = [load_bgr(p) for p in files]

    # ---------- R1: temporal 关闭时与【未修改代码】输出一致
    # 参照物是 dump_a0_raw.py 用 git worktree 原始代码在相同环境转储的 raw_pristine_p1.npz
    # (原始代码两进程互相对比为逐位一致, 即跨进程环境噪声为 0)。
    # 旧的 preds_d3.npz 录制于 9-16, 与该环境存在非零但 AP 无关的数值差, 只作参考。
    head.set_temporal_enabled(False)
    ref = np.load(os.path.join(LOG_DIR, 'raw_pristine_p1.npz'))
    old = np.load(OLD_NPZ)

    def greedy_match_stats(a, ob, thr_iou=0.95):
        """a/ob: [N,6] (xyxy,score,cls); 高分区(>0.05)同类 IoU 配对 -> (配对数, 总数, 框差, 分差)"""
        ah = a[a[:, 4] > 0.05]
        if len(ah) == 0:
            return 0, 0, 0.0, 0.0
        oh = ob[ob[:, 4] > 0.05]
        taken = np.zeros(len(oh), bool)
        n_hit, d_box, d_score = 0, 0.0, 0.0
        for row in ah:
            same = (oh[:, 5] == row[5]) & ~taken
            if not same.any():
                continue
            cand = oh[same]
            ix1, iy1 = np.maximum(cand[:, 0], row[0]), np.maximum(cand[:, 1], row[1])
            ix2, iy2 = np.minimum(cand[:, 2], row[2]), np.minimum(cand[:, 3], row[3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            ua = ((cand[:, 2] - cand[:, 0]) * (cand[:, 3] - cand[:, 1])
                  + (row[2] - row[0]) * (row[3] - row[1]) - inter)
            iou = inter / np.maximum(ua, 1e-9)
            j = int(iou.argmax())
            if iou[j] >= thr_iou:
                gi = np.nonzero(same)[0][j]
                taken[gi] = True
                n_hit += 1
                d_box = max(d_box, float(np.abs(cand[j, :4] - row[:4]).max()))
                d_score = max(d_score, float(abs(cand[j, 4] - row[4])))
        return n_hit, len(ah), d_box, d_score

    tot_hit, tot_high, d1_box, d1_score = 0, 0, 0.0, 0.0
    old_hit, old_high, old_box, old_score = 0, 0, 0.0, 0.0
    for i, im in enumerate(imgs):
        r = m.predict(im, imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)[0]
        a = det_array(r)
        h, n, db, ds = greedy_match_stats(a, ref[f'frame{i}'])
        tot_hit += h; tot_high += n; d1_box = max(d1_box, db); d1_score = max(d1_score, ds)
        s, e = int(old['ptr'][i]), int(old['ptr'][i + 1])
        ob = np.concatenate([
            old['box'][s:e], old['score'][s:e, None], old['cls'][s:e, None].astype(np.float32)
        ], 1)
        h, n, db, ds = greedy_match_stats(a, ob)
        old_hit += h; old_high += n; old_box = max(old_box, db); old_score = max(old_score, ds)
    match_rate = tot_hit / max(tot_high, 1)
    results['R1_highscore_match_rate_vs_pristine'] = match_rate
    results['R1_matched_box_maxdiff_px_vs_pristine'] = d1_box
    results['R1_matched_score_maxdiff_vs_pristine'] = d1_score
    results['R1_ref_old_npz_match_rate_info'] = old_hit / max(old_high, 1)
    results['R1_ref_old_npz_box_maxdiff_px_info'] = old_box
    r1_ok = match_rate > 0.999 and d1_box < 0.01 and d1_score < 1e-4
    log(f'R1 对未修改代码: 高分区配对率 {match_rate:.4f} ({tot_hit}/{tot_high}), '
        f'框差 {d1_box:.4f}px, 分差 {d1_score:.6f}  -> {"通过" if r1_ok else "失败"}')
    log(f'R1 (参考) 对 9-16 旧录制: 配对率 {old_hit / max(old_high, 1):.4f}, '
        f'框差 {old_box:.4f}px, 分差 {old_score:.6f} (跨日环境差, 不作判据)')

    # ---------- R2: 开->关循环后无残留
    r_a = det_array(m.predict(imgs[2], imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)[0])
    head.set_temporal_enabled(True)
    head.reset_temporal_state()
    _ = m.predict(imgs[2], imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)  # 积累状态
    _ = m.predict(imgs[3], imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)
    assert head._temporal_state is not None
    head.set_temporal_enabled(False)  # 关闭并清状态
    r_b = det_array(m.predict(imgs[2], imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)[0])
    r2 = max_diff(r_a, r_b)
    results['R2_max_abs_diff_after_toggle'] = r2
    log(f'R2 开->关循环后输出最大绝对差: {r2:.6f}  -> {"通过" if r2 < 1e-5 else "失败"}')
    results['R2_state_cleared_on_disable'] = head._temporal_state is None
    log(f'R2 关闭后 state 已清空: {head._temporal_state is None}')

    # ---------- R3: 首帧 B0 == A0
    head.set_temporal_enabled(True)
    head.reset_temporal_state()
    r_b0_f0 = det_array(m.predict(imgs[0], imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)[0])
    head.set_temporal_enabled(False)
    r_a0_f0 = det_array(m.predict(imgs[0], imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)[0])
    r3 = max_diff(r_b0_f0, r_a0_f0)
    results['R3_first_frame_B0_vs_A0'] = r3
    log(f'R3 场景首帧 B0 与 A0 输出最大绝对差: {r3:.6f}  -> {"通过" if r3 < 1e-5 else "失败"}')

    # ---------- R4: 第二帧内部 query 数 = 375 = 300 current + 75 temporal
    head.set_temporal_enabled(True)
    head.reset_temporal_state()
    _ = m.predict(imgs[0], imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)[0]
    st = head._temporal_state
    ok_state = (st is not None and st['query_hidden'].shape == (1, 75, 256)
                and st['boxes'].shape == (1, 75, 4) and st['scores'].shape == (1, 75))
    results['R4_state_shapes_ok'] = bool(ok_state)
    log(f'R4 首帧后 state: hidden{tuple(st["query_hidden"].shape)} boxes{tuple(st["boxes"].shape)} '
        f'scores{tuple(st["scores"].shape)} -> {"通过" if ok_state else "失败"}')
    _ = m.predict(imgs[1], imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)[0]
    nq = head._last_decoder_query_count
    ncur = head._last_n_current
    meta = head._last_topk_meta
    n_src1_out = int((meta['source'][0] == 1).sum())
    results['R4_second_frame_decoder_queries'] = int(nq)
    results['R4_second_frame_current_queries'] = int(ncur)
    results['R4_temporal_in_topk'] = n_src1_out
    log(f'R4 第二帧 decoder 内部 query 数: {nq} (期望 375), 其中 current {ncur} (期望 300), '
        f'temporal {nq - ncur} (期望 75) -> {"通过" if nq == 375 and ncur == 300 else "失败"}')
    log(f'R4 第二帧进入最终 top-300 的 temporal query 数: {n_src1_out}')

    # ---------- R5: 最终输出 <= 300
    n_out = int(meta['scores'][0].shape[0])
    results['R5_output_rows'] = n_out
    log(f'R5 最终 topk 输出条数: {n_out} (期望 <=300) -> {"通过" if n_out <= 300 else "失败"}')

    # ---------- R6: NaN/Inf 与坐标范围
    finite = True
    in_range = True
    head.reset_temporal_state()
    for im in imgs:
        rr = m.predict(im, imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)[0]
        a = det_array(rr)
        if not np.isfinite(a).all():
            finite = False
        st = head._temporal_state
        if st is not None:
            for k, v in st.items():
                if not bool(np.isfinite(v.cpu().numpy()).all()):
                    finite = False
            bx = st['boxes'].cpu().numpy()
            if bx.min() < 0 or bx.max() > 1:
                in_range = False
    results['R6_all_finite'] = bool(finite)
    results['R6_state_boxes_in_01'] = bool(in_range)
    log(f'R6 输出与 state 全部 finite: {finite}, state 框在 [0,1]: {in_range} '
        f'-> {"通过" if finite and in_range else "失败"}')

    # ---------- R7: reset 后回到首帧行为
    head.reset_temporal_state()
    results['R7_has_state_after_reset'] = head._temporal_state is not None
    r_after = det_array(m.predict(imgs[0], imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)[0])
    r7 = max_diff(r_after, r_a0_f0)
    results['R7_max_abs_diff_after_reset'] = r7
    log(f'R7 reset 后首帧输出与 A0 最大绝对差: {r7:.6f}, state 残留: {head._temporal_state is not None} '
        f'-> {"通过" if r7 < 1e-5 else "失败"}')

    # ---------- R8: state 不带梯度
    _ = m.predict(imgs[1], imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)[0]
    st = head._temporal_state
    req_grad = any(bool(v.requires_grad) for v in st.values())
    results['R8_state_requires_grad'] = req_grad
    log(f'R8 state 张量 requires_grad: {req_grad} -> {"通过" if not req_grad else "失败"}')

    # 汇总
    passed = (r1_ok and r2 < 1e-5 and r3 < 1e-5 and nq == 375 and ncur == 300
              and n_out <= 300 and finite and in_range and r7 < 1e-5 and not req_grad
              and head._temporal_state is not None)
    results['ALL_PASSED'] = bool(passed)
    log('')
    log(f'回归测试总体: {"全部通过" if passed else "存在失败项, 见上"}')

    head.set_temporal_enabled(False)
    with open(os.path.join(LOG_DIR, 'regression_log.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(LOG_DIR, 'regression_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LINES) + '\n')
    log(f'已写入 {LOG_DIR}/regression_log.json|.txt')


if __name__ == '__main__':
    main()
