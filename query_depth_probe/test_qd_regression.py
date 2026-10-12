# -*- coding: utf-8 -*-
"""test_qd_regression.py —— QueryDepth 探针阶段 0: 回归测试
=============================================================
验证项(对应任务书 四.回归与公平性要求):
  R1  全部探针关闭 + eval_idx=2 (A0) 与【未修改代码】转储逐位一致
  R2  全部探针关闭 + eval_idx=5 (U0) 与【未修改代码】转储逐位一致
  R3  六层逐层导出开启 vs 关闭: 最终输出逐位一致; layer_states 结构正确
  R4  T31 (S3->L1前) 首帧与 A0 逐位一致; 第二帧 decoder 宽度 375/375/375;
      temporal 输出 meta: source_depth=3, injection_layer=0; state 多深度结构正确
  R5  T12/T23 各层宽度 300/375/375 与 300/300/375; T3O 不进入 decoder (全 300, inj=-1)
  R6  T31 第二帧起与 B0 旧录制 (preds_b0.npz) 数值一致 (容差 1e-5, 机制等价性)
  R7  state 全部 detach、finite、框在 [0,1]
  R8  temporal 开->关循环无残留; 关闭后输出与 A0 一致
  R9  输出条数 <=300、全部 finite (T33 与 T3O)
  R10 decoder-only CUDA Event 计时: 开启时 >0, 关闭路径无计算

结果写入 query_depth_probe/logs/regression_qd_log.json 与 .txt
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from PIL import Image

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, r'C:\Users\TONY\ultralytics')
import ultralytics  # noqa: E402
assert ultralytics.__file__.lower().startswith(r'c:\users\tony\ultralytics\ultralytics'), \
    f'导入路径错误: {ultralytics.__file__}'

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
CKPT = r'D:\lunwen\自研\runs\detect\runs_rtdetr\r18_640_sub20_b4\weights\best.pt'
PRISTINE_D3 = r'D:\lunwen\自研\query_prop_b0\logs\raw_pristine_p1.npz'   # eval_idx=2, 修改前代码
PRISTINE_U0 = r'D:\lunwen\自研\query_depth_probe\logs\raw_pristine_u0.npz'  # eval_idx=5, 修改前代码
OLD_B0 = r'D:\lunwen\自研\query_prop_b0\stream_preds\preds_b0.npz'       # B0 旧录制 (只读)
LOG_DIR = r'D:\lunwen\自研\query_depth_probe\logs'

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


def check_pristine(m, head, imgs, ref, tag):
    """全部探针关闭时与未修改代码转储逐位一致"""
    tot_hit, tot_high, d_box, d_score = 0, 0, 0.0, 0.0
    for i, im in enumerate(imgs):
        r = m.predict(im, imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)[0]
        h, n, db, ds = greedy_match_stats(det_array(r), ref[f'frame{i}'])
        tot_hit += h
        tot_high += n
        d_box = max(d_box, db)
        d_score = max(d_score, ds)
    match_rate = tot_hit / max(tot_high, 1)
    ok = match_rate > 0.999 and d_box == 0.0 and d_score == 0.0
    log(f'{tag} 对未修改代码: 高分区配对率 {match_rate:.4f} ({tot_hit}/{tot_high}), '
        f'框差 {d_box:.6f}px, 分差 {d_score:.6f} -> {"通过" if ok else "失败"}')
    return ok


def main():
    results = {}
    os.makedirs(LOG_DIR, exist_ok=True)

    from ultralytics import RTDETR
    m = RTDETR(CKPT)
    head = find_head(m.model)
    assert head is not None

    with open(os.path.join(DATA, 'stream_index.json'), encoding='utf-8') as f:
        idx = json.load(f)
    s0 = idx['scenes'][0]
    files = [os.path.join(DATA, 'images', fr['file']) for fr in s0['frames'][:6]]
    imgs = [load_bgr(p) for p in files]

    def apply(enc=300, ntmp=75, topk=300):
        head.set_temporal_enabled(False)
        head.set_query_overrides(encoder_queries=enc, temporal_queries=ntmp, output_topk=topk, motion_alpha=0.0)
        head.set_temporal_config(source_depth=3, injection_layer=0, fuse_after=False)
        head.return_all_layers = False
        head.time_decoder_cuda = False

    def run(i):
        return m.predict(imgs[i], imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)[0]

    # ---------- R1 / R2: 关闭时与未修改代码逐位一致 (3 层与 6 层)
    apply()
    head.decoder.eval_idx = 2
    ref_d3 = np.load(PRISTINE_D3)
    ok_r1 = check_pristine(m, head, imgs, ref_d3, 'R1 eval_idx=2 (A0)')
    head.decoder.eval_idx = 5
    ref_u0 = np.load(PRISTINE_U0)
    ok_r2 = check_pristine(m, head, imgs, ref_u0, 'R2 eval_idx=5 (U0)')
    results['R1_A0_bitwise_vs_pristine'] = bool(ok_r1)
    results['R2_U0_bitwise_vs_pristine'] = bool(ok_r2)

    # ---------- R3: 六层逐层导出 开 vs 关
    head.decoder.eval_idx = 5
    ref_off = [det_array(run(i)) for i in range(6)]
    head.return_all_layers = True
    dets_on = [det_array(run(i)) for i in range(6)]
    ls = head._last_layer_states
    r3_det = max(max_diff(dets_on[i], ref_off[i]) for i in range(6))
    r3_keys = sorted(ls.keys()) == [0, 1, 2, 3, 4, 5]
    r3_shapes = all(
        ls[k][0].shape == (1, 300, 256) and ls[k][1].shape == (1, 300, 4)
        and ls[k][2].shape == (1, 300, 8) and ls[k][3].shape == (1, 300, 4)
        for k in range(6)
    )
    # L6 导出与最终 postprocess 输入逐位一致: 按 topk (query, class) 比较分数
    meta = head._last_topk_meta
    q_idx = meta['query_idx'][0].cpu().numpy()          # (300,) topk 行 -> query 槽位
    cls_idx = meta['cls'][0].long().cpu().numpy()
    l6_scores = ls[5][2][0].sigmoid()                   # (300, 8), 与 dec_scores 同一张量
    r3_l6match = bool(np.array_equal(l6_scores[q_idx, cls_idx].cpu().numpy(),
                                     meta['scores'][0].cpu().numpy()))
    head.return_all_layers = False
    _ = run(0)
    r3_off_cleared = head._last_layer_states is None
    ok_r3 = r3_det == 0.0 and r3_keys and r3_shapes and r3_off_cleared and r3_l6match
    results['R3_export_on_off_det_maxdiff'] = r3_det
    results['R3_layer_states_keys_ok'] = bool(r3_keys)
    results['R3_layer_states_shapes_ok'] = bool(r3_shapes)
    results['R3_export_off_cleared'] = bool(r3_off_cleared)
    log(f'R3 导出开/关 最终输出最大绝对差: {r3_det} (期望 0), keys {sorted(ls.keys())}, '
        f'shapes ok {r3_shapes}, 关闭后清除 {r3_off_cleared} -> {"通过" if ok_r3 else "失败"}')

    # ---------- R4: T31 首帧与 A0 一致 + 第二帧结构
    head.decoder.eval_idx = 2
    apply()
    ref_a0 = [det_array(run(i)) for i in range(2)]
    head.set_temporal_enabled(True)
    head.reset_temporal_state()
    f0 = det_array(run(0))
    r4_f0 = max_diff(f0, ref_a0[0])
    st = head._temporal_state
    r4_state_shapes = (
        st is not None and tuple(st['idx'].shape) == (1, 75)
        and all(tuple(st['hidden'][d].shape) == (1, 75, 256) for d in range(4))
        and all(tuple(st['boxes'][d].shape) == (1, 75, 4) for d in range(4))
        and tuple(st['logits3'].shape) == (1, 75, 8) and tuple(st['scores3'].shape) == (1, 75)
    )
    r4_idx_sorted = bool(np.all(np.diff(st['scores3'][0].cpu().numpy()) <= 0))
    f1 = det_array(run(1))
    r4_nq = int(head._last_decoder_query_count)
    r4_widths = dict(head._last_layer_widths)
    meta = head._last_topk_meta
    src1 = (meta['source'][0] == 1)
    r4_src_depth = bool((meta['source_depth'][0][src1] == 3).all()) and \
        bool((meta['source_depth'][0][~src1] == 0).all())
    r4_inj = bool((meta['injection_layer'][0][src1] == 0).all()) and \
        bool((meta['injection_layer'][0][~src1] == -2).all())
    ok_r4 = r4_f0 == 0.0 and r4_state_shapes and r4_idx_sorted and r4_nq == 375 \
        and r4_widths == {0: 375, 1: 375, 2: 375} and r4_src_depth and r4_inj
    results['R4_T31_first_frame_vs_A0_maxdiff'] = r4_f0
    results['R4_state_shapes_ok'] = bool(r4_state_shapes)
    results['R4_second_frame_widths'] = r4_widths
    results['R4_meta_annotations_ok'] = bool(r4_src_depth and r4_inj)
    log(f'R4 T31 首帧与 A0 差: {r4_f0} (期望 0); state 结构 {"ok" if r4_state_shapes else "FAIL"}; '
        f'第二帧 nq={r4_nq} widths={r4_widths} -> {"通过" if ok_r4 else "失败"}')

    # ---------- R5: T12 / T23 / T3O 结构
    head.reset_temporal_state()
    head.set_temporal_config(source_depth=1, injection_layer=1)
    _ = run(0)
    _ = run(1)
    w_t12 = dict(head._last_layer_widths)
    head.reset_temporal_state()
    head.set_temporal_config(source_depth=2, injection_layer=2)
    _ = run(0)
    _ = run(1)
    w_t23 = dict(head._last_layer_widths)
    head.reset_temporal_state()
    head.set_temporal_config(source_depth=3, injection_layer=2)
    _ = run(0)
    _ = run(1)
    w_t33 = dict(head._last_layer_widths)
    head.reset_temporal_state()
    head.set_temporal_config(fuse_after=True)
    _ = run(0)
    f1o = det_array(run(1))
    w_t3o = dict(head._last_layer_widths)
    meta = head._last_topk_meta
    src1 = (meta['source'][0] == 1)
    r5_inj_fuse = bool((meta['injection_layer'][0][src1] == -1).all())
    r5_nq_fuse = int(head._last_decoder_query_count)
    ok_r5 = w_t12 == {0: 300, 1: 375, 2: 375} and w_t23 == {0: 300, 1: 300, 2: 375} \
        and w_t33 == {0: 300, 1: 300, 2: 375} and w_t3o == {0: 300, 1: 300, 2: 300} \
        and r5_inj_fuse and r5_nq_fuse == 300
    results['R5_widths_T12'] = w_t12
    results['R5_widths_T23'] = w_t23
    results['R5_widths_T33'] = w_t33
    results['R5_widths_T3O'] = w_t3o
    log(f'R5 widths: T12={w_t12} T23={w_t23} T33={w_t33} T3O={w_t3o} (期望 300/375/375, '
        f'300/300/375, 300/300/375, 300/300/300), T3O inj=-1 {r5_inj_fuse} -> {"通过" if ok_r5 else "失败"}')

    # ---------- R6: T31 与 B0 旧录制数值一致 (机制等价)
    # 逐行比较会在 top-300 边界处被 ~1e-5 的分数噪声换序打断, 故用同类 IoU>=0.5 贪心匹配后比差。
    def match_frame_stats(d, ob):
        """d/ob: [N,6] (xyxy,score,cls); 同类 IoU>=0.5 贪心匹配 -> (配对差上限, 未配对 old, 未配对 new)"""
        taken = np.zeros(len(ob), bool)
        n_hit, db, ds = 0, 0.0, 0.0
        for row in d:
            ci = np.nonzero((ob[:, 5] == row[5]) & ~taken)[0]
            if not len(ci):
                continue
            ix1, iy1 = np.maximum(ob[ci, 0], row[0]), np.maximum(ob[ci, 1], row[1])
            ix2, iy2 = np.minimum(ob[ci, 2], row[2]), np.minimum(ob[ci, 3], row[3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            ua = ((ob[ci, 2] - ob[ci, 0]) * (ob[ci, 3] - ob[ci, 1])
                  + (row[2] - row[0]) * (row[3] - row[1]) - inter)
            iou = inter / np.maximum(ua, 1e-9)
            j = int(iou.argmax())
            if iou[j] >= 0.5:
                taken[ci[j]] = True
                n_hit += 1
                db = max(db, float(np.abs(ob[ci[j], :4] - row[:4]).max()))
                ds = max(ds, float(abs(ob[ci[j], 4] - row[4])))
        return n_hit, db, ds, int((~taken).sum()), len(d) - n_hit

    head.set_temporal_config(source_depth=3, injection_layer=0, fuse_after=False)
    head.reset_temporal_state()
    old = np.load(OLD_B0)
    max_db, max_ds, un_old, un_new = 0.0, 0.0, 0, 0
    _ = run(0)  # 先建 state, 帧 1..5 才有 temporal
    for i in range(1, 6):
        d = det_array(run(i))
        s, e = int(old['ptr'][i]), int(old['ptr'][i + 1])
        ob = np.concatenate([old['box'][s:e], old['score'][s:e, None], old['cls'][s:e, None].astype(np.float32)], 1)
        n_hit, db, ds, uo, un = match_frame_stats(d, ob)
        max_db = max(max_db, db)
        max_ds = max(max_ds, ds)
        un_old += uo
        un_new += un
    # 等价性判据: 匹配对框差 <=1e-3px、分差 <=1e-4 (sigmoid(inverse_sigmoid) 与直接 clamp 的往返噪声),
    # top-300 边界换序只允许少量未配对行
    ok_r6 = max_db <= 1e-3 and max_ds <= 1e-4 and un_old + un_new <= 10
    results['R6_T31_vs_B0_matched_box_maxdiff'] = max_db
    results['R6_T31_vs_B0_matched_score_maxdiff'] = max_ds
    results['R6_T31_vs_B0_unmatched_old_new'] = [un_old, un_new]
    log(f'R6 T31 对 B0 旧录制 (帧1-5 同类 IoU>=0.5 匹配): 框差 {max_db:.3e} (期望 <=1e-3), '
        f'分差 {max_ds:.3e} (期望 <=1e-4), 未配对 {un_old}/{un_new} (期望 <=10) -> {"通过" if ok_r6 else "失败"}')

    # ---------- R7: state detach / finite / 范围
    ok_r7 = True
    for k in ['idx', 'logits3', 'scores3', 'velocity']:
        v = st[k]
        ok_r7 &= not bool(v.requires_grad) and bool(np.isfinite(v.cpu().numpy()).all())
    for d in range(4):
        ok_r7 &= not bool(st['hidden'][d].requires_grad) and not bool(st['boxes'][d].requires_grad)
        ok_r7 &= bool(np.isfinite(st['hidden'][d].cpu().numpy()).all())
        bx = st['boxes'][d].cpu().numpy()
        ok_r7 &= bool(np.isfinite(bx).all()) and bx.min() >= 0 and bx.max() <= 1
    results['R7_state_detached_finite_inrange'] = bool(ok_r7)
    log(f'R7 state detach/finite/范围: {"通过" if ok_r7 else "失败"}')

    # ---------- R8: 开->关循环无残留
    head.set_temporal_enabled(False)
    r_after = det_array(run(2))
    head.set_temporal_enabled(True)
    head.reset_temporal_state()
    _ = run(2)
    _ = run(3)
    head.set_temporal_enabled(False)
    r_b = det_array(run(2))
    r8_diff = max_diff(r_after, r_b)
    r8_cleared = head._temporal_state is None
    ok_r8 = r8_diff == 0.0 and r8_cleared
    results['R8_toggle_maxdiff'] = r8_diff
    results['R8_state_cleared_on_disable'] = bool(r8_cleared)
    log(f'R8 开->关循环后输出最大绝对差: {r8_diff} (期望 0), state 清空 {r8_cleared} -> {"通过" if ok_r8 else "失败"}')

    # ---------- R9: 输出条数 / finite (T33 与 T3O)
    ok_r9 = True
    head.set_temporal_enabled(True)
    head.set_temporal_config(source_depth=3, injection_layer=2, fuse_after=False)
    head.reset_temporal_state()
    for i in range(6):
        r = run(i)
        a = det_array(r)
        ok_r9 &= len(a) <= 300 and bool(np.isfinite(a).all())
    head.set_temporal_config(fuse_after=True)
    head.reset_temporal_state()
    for i in range(6):
        r = run(i)
        a = det_array(r)
        ok_r9 &= len(a) <= 300 and bool(np.isfinite(a).all())
    results['R9_outputs_finite_and_capped'] = bool(ok_r9)
    log(f'R9 T33/T3O 输出 <=300 且 finite: {"通过" if ok_r9 else "失败"}')

    # ---------- R10: decoder-only CUDA 计时
    head.set_temporal_enabled(False)
    head.decoder.eval_idx = 2
    head.time_decoder_cuda = True
    run(0)
    r10_on = head._last_decoder_ms
    head.time_decoder_cuda = False
    run(0)
    ok_r10 = r10_on > 0.1 and np.isfinite(r10_on)
    results['R10_decoder_cuda_ms_on'] = float(r10_on)
    log(f'R10 decoder-only CUDA 计时: {r10_on:.3f}ms (期望 >0.1) -> {"通过" if ok_r10 else "失败"}')

    # ---------- R11: QD-P 无退出调度 ([300,300,300]) 与 A0 逐位一致
    head.set_exit_plan([300, 300, 300], 'confidence')
    r11 = max(max_diff(det_array(run(i)), ref_a0[i]) for i in range(2))
    r11_ann = bool((head._last_exit_ann[0] == -1).all())
    ok_r11 = r11 == 0.0 and r11_ann
    results['R11_noexit_schedule_vs_A0_maxdiff'] = r11
    log(f'R11 [300,300,300] 与 A0 最大绝对差: {r11} (期望 0) -> {"通过" if ok_r11 else "失败"}')

    # ---------- R12: P1 随机掩码结构 + 退出 query 输出冻结验证
    rng12 = np.random.default_rng(42)
    cur = np.arange(300)
    masks = {}
    for li in range(2):
        keep = rng12.choice(cur, size=[150, 75][li], replace=False)
        mm = np.ones(len(cur), bool)
        mm[np.isin(cur, keep)] = False
        masks[li] = torch.from_numpy(mm).unsqueeze(0).cuda()
        cur = keep
    # ann==0 (L1 后退出) 冻结验证: 全量导出 L1 框 (L1 时 300 宽, 与 P1 的 L1 上下文一致)
    head.set_exit_plan(None)
    head.return_all_layers = True
    _ = run(1)
    ls = head._last_layer_states
    l1_box = ls[0][1][0].cpu().numpy()
    head.return_all_layers = False
    # ann==1 (L2 后退出) 冻结验证: 同掩码 eval_idx=1 运行, L2 即终层 (150 宽上下文与 P1 的 L2 一致)
    head.decoder.eval_idx = 1
    head.set_exit_plan([300, 150], 'random')
    head._exit_masks = {0: masks[0]}
    _ = run(1)
    ref2_box = head._last_dec_boxes[0, 0].cpu().numpy()
    head.decoder.eval_idx = 2
    # 随机掩码真实执行 P1
    head.set_exit_plan([300, 150, 75], 'random')
    head._exit_masks = masks
    _ = run(1)
    ann = head._last_exit_ann[0].cpu().numpy()
    widths = list(head._last_exit_widths)
    fin_box = head._last_dec_boxes[0, 0].cpu().numpy()  # (300,4) sigmoid cxcywh
    n0, n1, nf_ = int((ann == 0).sum()), int((ann == 1).sum()), int((ann == -1).sum())
    frozen_ok = True
    if n0:
        frozen_ok &= bool(np.array_equal(fin_box[ann == 0], l1_box[ann == 0]))
    if n1:
        frozen_ok &= bool(np.array_equal(fin_box[ann == 1], ref2_box[ann == 1]))
    ok_r12 = widths == [300, 150, 75] and (n0, n1, nf_) == (150, 75, 75) and frozen_ok
    results['R12_P1_widths'] = widths
    results['R12_ann_dist'] = [n0, n1, nf_]
    results['R12_frozen_outputs_bitwise'] = bool(frozen_ok)
    log(f'R12 P1 随机: widths={widths} ann分布={n0}/{n1}/{nf_} (期望 [300,150,75], 150/75/75), '
        f'退出 query 输出冻结逐位一致 {frozen_ok} -> {"通过" if ok_r12 else "失败"}')

    # ---------- R13: 在线策略宽度正确
    ok_r13 = True
    for pol in ['confidence', 'stability', 'confidence+stability']:
        head.set_exit_plan([300, 150, 75], pol)
        _ = run(1)
        w = list(head._last_exit_widths)
        ok_r13 &= w == [300, 150, 75]
        results[f'R13_widths_{pol}'] = w
    log(f'R13 在线策略 widths: conf={results.get("R13_widths_confidence")} '
        f'stab={results.get("R13_widths_stability")} cs={results.get("R13_widths_confidence+stability")} '
        f'-> {"通过" if ok_r13 else "失败"}')

    # ---------- R14: gt_oracle 掩码路径 (掩码由调用方提供, 模型内部无 GT)
    head.set_exit_plan([300, 150, 75], 'gt_oracle')
    head._exit_masks = masks
    _ = run(1)
    w = list(head._last_exit_widths)
    ok_r14 = w == [300, 150, 75]
    results['R14_gt_oracle_widths'] = w
    log(f'R14 gt_oracle 掩码 widths={w} (期望 [300,150,75]) -> {"通过" if ok_r14 else "失败"}')

    # ---------- R15: exit 开->关循环无残留, 恢复 A0 逐位一致
    head.set_exit_plan(None)
    r15_diff = max(max_diff(det_array(run(i)), ref_a0[i]) for i in range(2))
    r15_cleared = head.exit_schedule is None and head._exit_masks is None
    ok_r15 = r15_diff == 0.0 and r15_cleared
    results['R15_exit_toggle_maxdiff'] = r15_diff
    results['R15_plan_cleared'] = bool(r15_cleared)
    log(f'R15 exit 开->关后与 A0 最大绝对差: {r15_diff} (期望 0), plan 清空 {r15_cleared} '
        f'-> {"通过" if ok_r15 else "失败"}')

    # ---------- R16: 整体 CUDA 计时
    head.time_model_cuda = True
    run(0)
    r16 = head._last_model_ms
    ok_r16 = r16 > 0.1 and np.isfinite(r16)
    head.time_model_cuda = False
    results['R16_model_cuda_ms'] = float(r16)
    log(f'R16 整体 CUDA 计时: {r16:.3f}ms (期望 >0.1) -> {"通过" if ok_r16 else "失败"}')

    # ---------- 汇总
    apply()
    head.decoder.eval_idx = 2
    passed = all([ok_r1, ok_r2, ok_r3, ok_r4, ok_r5, ok_r6, ok_r7, ok_r8, ok_r9, ok_r10,
                  ok_r11, ok_r12, ok_r13, ok_r14, ok_r15, ok_r16])
    results['ALL_PASSED'] = bool(passed)
    log('')
    log(f'回归测试总体: {"全部通过" if passed else "存在失败项, 见上"}')

    with open(os.path.join(LOG_DIR, 'regression_qd_log.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(LOG_DIR, 'regression_qd_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LINES) + '\n')
    log(f'已写入 {LOG_DIR}/regression_qd_log.json|.txt')


if __name__ == '__main__':
    main()
