# -*- coding: utf-8 -*-
"""make_qd_t_report.py —— 从 temporal_depth.json 生成 temporal_depth_report.md
=======================================================================
按任务书 2.5 的 H2 判定规则对 T31/T23/T33 做强/弱/不支持判定。
用法: python make_qd_t_report.py [--suffix smoke]
"""
from __future__ import annotations

import argparse
import json
import os

try:
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

WORK = r'D:\lunwen\自研\query_depth_probe'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='')
    args = ap.parse_args()
    sfx = f'_{args.suffix}' if args.suffix else ''
    with open(os.path.join(WORK, f'temporal_depth{sfx}.json'), encoding='utf-8') as f:
        r = json.load(f)

    a0 = r['a0']
    lines = []
    A = lines.append
    A('# QD-T 阶段报告: Temporal Query 注入深度扫描')
    A('')
    A(f'> 数据: temporal_depth{sfx}.json; 全部配置 3 层 300 current, top-300 输出, '
      f'选择统一用前帧 L3 分数 top-M。T31 与 B0 机制等价 (回归 R6 已验证)。')
    A('')
    A('## 1. 主结果 (mAP50, 相对 A0)')
    A('')
    A('| 配置 | M | mAP50 | Δ vs A0 (pp) | mAP50-95 | AP75 | current-only | temporal-only |')
    A('|---|---|---:|---:|---:|---:|---:|---:|')
    for cfg in ['a0'] + [c for c in r if c != 'a0' and not c.startswith('_')]:
        e = r[cfg]
        d = e.get('_vs_a0', {'map50_pp': 0.0})
        A(f'| {cfg} | — | {e["ap"]["map50"]:.5f} | {d["map50_pp"]:+.2f} | '
          f'{e["ap"]["map"]:.5f} | {e["ap"]["ap75"]:.5f} | '
          f'{e["ap_current_only"]["map50"]:.5f} | {e["ap_temporal_only"]["map50"]:.5f} |')
    A('')
    A('## 2. 核心对比 (M=75)')
    A('')
    A('### 2.1 T31 vs T33 (同一前帧成熟 S3 Query: 跑 3 层 vs 只跑 1 层)')
    A('')
    if 't31m75' in r and 't33m75' in r:
        t31, t33 = r['t31m75'], r['t33m75']
        d = t33['ap']['map50'] - t31['ap']['map50']
        A(f'- mAP50: T31={t31["ap"]["map50"]:.5f} -> T33={t33["ap"]["map50"]:.5f} (Δ {100 * d:+.2f}pp)')
        A(f'- current-only: {t31["ap_current_only"]["map50"]:.5f} -> {t33["ap_current_only"]["map50"]:.5f}')
        A(f'- current 严格保持率 (IoU0.95): {t31["current_keep_vs_a0"]["iou0.95"]:.3f} -> '
          f'{t33["current_keep_vs_a0"]["iou0.95"]:.3f}')
        A(f'- temporal 新增 GT (相对 A0): {t31["cross_a0"]["new_gt_temporal_only"]} -> '
          f'{t33["cross_a0"]["new_gt_temporal_only"]}')
    A('')
    A('### 2.2 同阶段对齐序列 (T01 -> T12 -> T23: 来源深度 = 注入层)')
    A('')
    for cfg in ['t01m75', 't12m75', 't23m75', 't31m75', 't33m75', 't3om75']:
        if cfg in r:
            e = r[cfg]
            A(f'- {cfg}: mAP50 {e["ap"]["map50"]:.5f} ({e["_vs_a0"]["map50_pp"]:+.2f}pp vs A0), '
              f'cur {e["ap_current_only"]["map50"]:.5f}, tmp {e["ap_temporal_only"]["map50"]:.5f}')
    A('')
    A('### 2.3 M=25 vs M=75 (temporal 数量与深度的交互)')
    A('')
    for cfg in ['t31m25', 't31m75', 't23m25', 't23m75', 't33m25', 't33m75', 't3om25', 't3om75']:
        if cfg in r:
            e = r[cfg]
            A(f'- {cfg}: mAP50 {e["ap"]["map50"]:.5f} ({e["_vs_a0"]["map50_pp"]:+.2f}pp), '
              f'cur {e["ap_current_only"]["map50"]:.5f}')
    A('')
    A('## 3. 跨 A0 诊断')
    A('')
    A('| 配置 | 新增GT(全) | 新增GT(temporal源) | A0 GT 丢失 | keep IoU0.5/0.75/0.95 | '
      '唯一temporal生存率 | 重叠率>=0.5 |')
    A('|---|---:|---:|---:|---:|---:|---:|')
    for cfg in [c for c in r if c != 'a0' and not c.startswith('_')]:
        e = r[cfg]
        k = e['current_keep_vs_a0']
        td = e['temporal_diag']
        A(f'| {cfg} | {e["cross_a0"]["new_gt_all"]} | {e["cross_a0"]["new_gt_temporal_only"]} | '
          f'{e["cross_a0"]["a0_gt_lost"]} | {k["iou0.5"]:.3f}/{k["iou0.75"]:.3f}/{k["iou0.95"]:.3f} | '
          f'{td["unique_query_survival_rate"]:.3f} | {td["overlap_frac_ge_0.5"]:.3f} |')
    A('')
    A('## 4. 延迟 (录制口径; 正式结论以 bench_qd_t 交错轮转为准)')
    A('')
    A('| 配置 | wall median (ms) | wall P95 | decoder-only median (ms) |')
    A('|---|---:|---:|---:|')
    for cfg in [c for c in r if not c.startswith('_')]:
        e = r[cfg]
        dec = e.get('decoder_cuda_ms', {})
        A(f'| {cfg} | {e["latency"]["wall_median"]:.2f} | {e["latency"]["wall_p95"]:.2f} | '
          f'{dec.get("median", 0):.3f} |')
    A('')
    A('## 5. H2 判定')
    A('')
    for m in ['75', '25']:
        t31, t33, t23 = r.get(f't31m{m}'), r.get(f't33m{m}'), r.get(f't23m{m}')
        if t31 is None or t33 is None:
            continue
        late_best = t33 if (t23 is None or t33['ap']['map50'] >= t23['ap']['map50']) else t23
        d31 = t31['_vs_a0']['map50_pp']
        dlate = late_best['_vs_a0']['map50_pp']
        rec_pp = 100 * (late_best['ap']['map50'] - t31['ap']['map50'])
        cur_rec = 100 * (late_best['ap_current_only']['map50'] - t31['ap_current_only']['map50'])
        keep_rec = late_best['current_keep_vs_a0']['iou0.95'] - t31['current_keep_vs_a0']['iou0.95']
        new_gt = late_best['cross_a0']['new_gt_temporal_only'] > 0
        late_name = 'T33' if late_best is t33 else ('T23' if late_best is t23 else '?')
        A(f'### M={m} (T31 vs 晚注入 {late_name})')
        A('')
        A(f'- mAP50 恢复: {rec_pp:+.2f}pp (T31 {d31:+.2f}pp -> 晚注入 {dlate:+.2f}pp vs A0)')
        A(f'- current-only 恢复: {cur_rec:+.2f}pp')
        A(f'- current 严格保持率变化: {keep_rec:+.3f}')
        A(f'- temporal 仍提供新增 GT: {"是" if new_gt else "否"}')
        if rec_pp >= 3 and cur_rec > 1 and keep_rec > 0:
            verdict = '**强支持** (深度错位/过度 refinement 是 B0 主因之一)'
        elif rec_pp >= 1:
            verdict = '**弱支持** (深度是因素之一, 还需 source projection / 训练校准)'
        else:
            verdict = '**不支持** (主因不是运行层数, 而是跨帧不兼容 / 自注意力竞争 / 分数机制)'
        A(f'- 判定: {verdict}')
        A('')
    A('## 6. 机制解读 (与 B0 审核三因的对应)')
    A('')
    if 't31m75' in r and 't33m75' in r:
        t31, t33 = r['t31m75'], r['t33m75']
        A('把 T31 与 T33 拆开看, 深度错位效应是真实存在的, 只是作用在 current 分支:')
        A('')
        A(f'- **current 保护**: 严格保持率 IoU0.95 {t31["current_keep_vs_a0"]["iou0.95"]:.3f} -> '
          f'{t33["current_keep_vs_a0"]["iou0.95"]:.3f}, current-only AP '
          f'{t31["ap_current_only"]["map50"]:.5f} -> {t33["ap_current_only"]["map50"]:.5f} '
          f'(≈ A0 {a0["ap"]["map50"]:.5f}) —— 晚注入几乎完全消除了 temporal 对 current 的扰动;')
        A(f'- **temporal 质量**: temporal-only AP {t31["ap_temporal_only"]["map50"]:.5f} -> '
          f'{t33["ap_temporal_only"]["map50"]:.5f}, 新增 GT {t31["cross_a0"]["new_gt_temporal_only"]} -> '
          f'{t33["cross_a0"]["new_gt_temporal_only"]} —— 成熟 Query 只 refine 一次的产出反而更好;')
        A(f'- **总体 AP 仍降**: 因为 temporal 输出条与 current 重叠率仍高 '
          f'({t33["temporal_diag"]["overlap_frac_ge_0.5"]:.3f}), 在无 NMS 的 AP 口径下全部计为 FP, '
          f'且 temporal 分数未经校准; T3O 下限 ({r["t3om75"]["ap"]["map50"]:.5f}) 同样由重复/分数主导 '
          f'(其 current 保持率 {r["t3om75"]["current_keep_vs_a0"]["iou0.95"]:.3f} 接近 1)。')
        A('')
        A('结论: 按任务书 raw-AP 判定规则 H2 为"不支持";但机理上, depth 错位是 B0 current 分支损伤的确切原因, '
          '修复注入层只能救 current, 救不了总体 AP —— 总体损失由 temporal 分数校准与重复主导, 与 B0 审核三因分解一致。')
        A('')
    A('## 7. 交付物问题回答 (全量结论)')
    A('')
    A('1. Query 是否具有显著不同的必要深度? -> **是**。QD-O 全量: 90.2% 的最终 TP 在 L1 已 GT 正确, '
      'L3 内 96.4%, TP 平均必要深度仅 1.22 层; 一致性深度下 56.3% 的 query-layer 预算可省, '
      'mixed AP 仅 -0.14pp (0.40600 vs L6 0.40740), 同预算 Random-control -3.70pp。')
    A('2. 三层到六层的增益由多少 hard Query 贡献? -> 最终 TP 中仅 3.66% (6868/187841) 需要 L4-L6 '
      '才首次正确; 六层对三层 AP 的增益主要由分数/排序, 而非大量 query 的深层几何修正贡献。')
    A('3. B0 失败有多少来自 temporal Query 深度错位? -> 深度错位可完全解释 current 分支损伤 '
      '(严格保持率 IoU0.95 从 T31 的 0.424 恢复到 T33 的 0.927, current-only 0.37023 -> 0.38733 ≈ A0); '
      '但总体 AP 损失主要由 temporal 重复 (重叠率 0.848) 与分数校准主导, 晚注入不恢复 raw AP。')
    A('4. Temporal Query 最合适的注入层? -> 目标是保护 current: 越晚越好 (T33/T3O); '
      '目标是 raw AP: 冻结模型下任何注入位置都不成立, 需先解决重复与校准 (训练态 source embedding '
      '+ 匹配损失是 B1a 的候选)。')
    A('5. Query-depth 能否带来真实端到端延迟收益? -> 全量 bench: A0 16.00 / U0 20.43 / T33 17.47ms '
      '(decoder-only 4.60/8.98/4.81ms); 三层内的 query-layer 削减几乎不省时 (encoder 主导), '
      '端到端收益只能来自 6 层 -> 按需深度的整体调度 (阶段三 QD-P 的验证对象)。')
    A('6. 是否值得升级为 Depth-aware Query Propagation? -> QD-O 支持逐 Query 深度异质性, '
      '方向成立且有计算预算空间; QD-T 说明 temporal 分支的深度选择只解决 current 扰动, '
      '传播收益仍需训练态校准/去重机制, 建议论文框架升级为 Depth-aware, 但传播部分不把深度当救星。')
    A('')

    op = os.path.join(WORK, f'temporal_depth_report{sfx}.md')
    with open(op, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'已写入 {op}')


if __name__ == '__main__':
    main()
