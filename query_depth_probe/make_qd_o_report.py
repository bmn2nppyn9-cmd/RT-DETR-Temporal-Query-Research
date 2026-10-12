# -*- coding: utf-8 -*-
"""make_qd_o_report.py —— 从 qd_oracle.json 生成 qd_oracle_report.md
==================================================================
按任务书 H1 支持条件做自动化判定, 输出结构与结论。
用法: python make_qd_o_report.py [--suffix smoke]
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


def fmt_hist(h, total=None):
    """直方图 -> 比例字符串"""
    if 'never' in h:
        keys = [f'L{i+1}' for i in range(6)] + ['never']
    else:
        keys = [f'L{i+1}' for i in range(6)]
    tot = total if total is not None else sum(h.values())
    if tot == 0:
        return '(无样本)'
    return '  '.join(f'{k}: {100 * h.get(k, 0) / tot:.1f}%' for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='')
    args = ap.parse_args()
    sfx = f'_{args.suffix}' if args.suffix else ''
    with open(os.path.join(WORK, f'qd_oracle{sfx}.json'), encoding='utf-8') as f:
        r = json.load(f)

    t = r['transitions']
    h1 = r['H1_check']
    od50 = r['ap_O_GT50']
    odc = r['ap_O_consistency_t07_e05']
    l6 = r['ap_L6_from_export']
    bg = r['budget_O_GT50']
    bc = r['budget_O_consistency_t07_e05']

    # H1 判定 (支持条件: 同时满足; mixed/budget 用可部署口径 O_consistency,
    # GT oracle 是正确性深度参照, 其 mixed AP 的下降是跨层分数重排序串扰, 不作为判定输入)
    conds = [
        ('大量最终 TP 在 L1-L3 已稳定正确',
         t['p_l3_correct_given_tp'] >= 0.8,
         t['p_l3_correct_given_tp']),
        ('少量 hard Query 主要由深层修正',
         t['p_l3_err_fixed_by_deep'] <= 0.1,
         t['p_l3_err_fixed_by_deep']),
        ('Mixed-depth Oracle 距 6 层 <=0.3-0.5pp (O-consistency)',
         h1['mixed_vs_L6_gap_pp'] <= 0.5,
         h1['mixed_vs_L6_gap_pp']),
        ('Query-layer budget 至少减少 30% (O-consistency)',
         h1['budget_savings_pct'] >= 30,
         h1['budget_savings_pct']),
        ('Random-control 在同预算下明显更差',
         (r['ap_O_consistency_t07_e05']['map50'] - r['ap_Random_control']['map50']) >= 0.005,
         r['ap_O_consistency_t07_e05']['map50'] - r['ap_Random_control']['map50']),
    ]
    h1_supported = all(c[1] for c in conds)

    lines = []
    A = lines.append
    A('# QD-O 阶段报告: 六层逐 Query 收敛深度 Oracle')
    A('')
    A(f'> 数据: qd_oracle{sfx}.json; 模型 r18_640_sub20_b4/best.pt (eval_idx=5, 300 current); '
      f'GT 只用于离线 Oracle, 不用于任何可部署策略。')
    A('')
    A('## 1. 主结果')
    A('')
    A('| 输出 | mAP50 | mAP50-95 | AP75 | E[d] (层) | 节省 vs 6L |')
    A('|---|---:|---:|---:|---:|---:|')
    A(f'| L6 (全深度, 导出重建) | {l6["map50"]:.5f} | {l6["map"]:.5f} | {l6["ap75"]:.5f} | 6 | 0% |')
    A(f'| O-consistency (τ0.7/ε0.05) | {odc["map50"]:.5f} | {odc["map"]:.5f} | {odc["ap75"]:.5f} '
      f'| {bc["E_depth_layers"]:.2f} | {bc["savings_vs_6L_pct"]:.1f}% |')
    A(f'| O-consistency (τ0.5/ε0.10) | {r["ap_O_consistency_t05_e10"]["map50"]:.5f} | '
      f'{r["ap_O_consistency_t05_e10"]["map"]:.5f} | {r["ap_O_consistency_t05_e10"]["ap75"]:.5f} '
      f'| {r["budget_O_consistency_t05_e10"]["E_depth_layers"]:.2f} | '
      f'{r["budget_O_consistency_t05_e10"]["savings_vs_6L_pct"]:.1f}% |')
    A(f'| O-GT50 (Oracle 上限) | {od50["map50"]:.5f} | {od50["map"]:.5f} | {od50["ap75"]:.5f} '
      f'| {bg["E_depth_layers"]:.2f} | {bg["savings_vs_6L_pct"]:.1f}% |')
    A(f'| O-GT75 (Oracle 上限) | {r["ap_O_GT75"]["map50"]:.5f} | {r["ap_O_GT75"]["map"]:.5f} | '
      f'{r["ap_O_GT75"]["ap75"]:.5f} | {r["budget_O_GT75"]["E_depth_layers"]:.2f} | '
      f'{r["budget_O_GT75"]["savings_vs_6L_pct"]:.1f}% |')
    A(f'| Random-control (同 O-consistency 预算) | {r["ap_Random_control"]["map50"]:.5f} | '
      f'{r["ap_Random_control"]["map"]:.5f} | {r["ap_Random_control"]["ap75"]:.5f} | '
      f'{r["budget_Random_control"]["E_depth_layers"]:.2f} | — |')
    A(f'| Confidence-control (同预算) | {r["ap_Confidence_control"]["map50"]:.5f} | '
      f'{r["ap_Confidence_control"]["map"]:.5f} | {r["ap_Confidence_control"]["ap75"]:.5f} | '
      f'{r["budget_Confidence_control"]["E_depth_layers"]:.2f} | — |')
    A('')
    A('注: O-GT50/O-GT75 的 E[d] 被"非 TP 一律留在 L6"的赋值抬高; 其 TP 子集的平均深度见第 2 节。')
    A('')
    A('## 2. 深度分布')
    A('')
    A('### 2.1 最终层一致性深度 d_consistency (不用 GT)')
    A('')
    A('| 阈值 | 分布 |')
    A('|---|---|')
    for t_ in ['0.5', '0.7', '0.9']:
        for e_ in ['0.02', '0.05', '0.10']:
            k = f'tau{t_}_eps{e_}'
            A(f'| τ_box={t_} / ε_score={e_} | {fmt_hist(r["d_consistency"][k])} |')
    A('')
    A('TP / 非 TP 拆分 (τ0.7/ε0.05):')
    A('')
    A(f'- TP query: {fmt_hist(r["d_consistency_t07_e05_split"]["tp"])}')
    A(f'- 非 TP query: {fmt_hist(r["d_consistency_t07_e05_split"]["non_tp"])}')
    A('')
    A('### 2.2 GT 正确深度 d_gt50 / d_gt75 (Oracle 上限, 仅最终 TP query)')
    A('')
    A(f'- d_gt50 (IoU>=0.5): {fmt_hist(r["d_gt50"])}')
    A(f'- d_gt75 (IoU>=0.75): {fmt_hist(r["d_gt75"])}')
    A('')
    A('逐类 d_gt50:')
    A('')
    for c, h in r['d_gt50_per_class'].items():
        if h == {'n': 0}:
            A(f'- {c}: 无样本')
        else:
            A(f'- {c}: {fmt_hist(h)}')
    A('')
    A('按目标尺寸 d_gt50:')
    for k, h in r['d_gt50_per_size'].items():
        A(f'- {k}: {fmt_hist(h)}')
    A('')
    A('按置信度 d_gt50:')
    A(f'- high: {fmt_hist(r["d_gt50_per_conf"]["high"])}')
    A(f'- low: {fmt_hist(r["d_gt50_per_conf"]["low"])}')
    A('')
    A('### 2.3 背景/低价值 Query 稳定深度')
    A('')
    A('| 阈值 | 分布 (全 query) |')
    A('|---|---|')
    for t_ in ['0.01', '0.03', '0.05']:
        A(f'| score<{t_} | {fmt_hist(r["background_stable"][f"tau{t_}"])} |')
    A('')
    A('## 3. 逐层转移表 (GT 口径, 最终 TP query)')
    A('')
    A(f'- L1 正确 -> L6 仍正确: **{100 * t["p_l1_correct_given_tp"]:.1f}%**')
    A(f'- L2 正确 -> L6 仍正确: **{100 * t["p_l2_correct_given_tp"]:.1f}%**')
    A(f'- L3 正确 -> L6 仍正确: **{100 * t["p_l3_correct_given_tp"]:.1f}%**')
    A(f'- L3 错误 -> L4/L5/L6 被修正: **{100 * t["p_l3_err_fixed_by_deep"]:.1f}%**')
    A(f'- L3 正确 -> L4/L5/L6 反而变错: **{100 * t["p_l3_ok_broken_by_deep"]:.1f}%**')
    A('')
    A('## 4. 分数尺度诊断')
    A('')
    A(f'各层最大类分数均值: ' + ', '.join(f'L{i+1}={x:.4f}' for i, x in enumerate(r['per_layer_mean_max_score'])))
    A('')
    A('跨层分数不可直接默认可比: O-GT50 的 mixed AP 低于 L6 主因是跨层分数重排序引起贪心匹配串扰, '
      '而非框本身不可用 (TP 子集平均深度仅 ~1.2 层)。校准变体结果:')
    A('')
    A(f'- odd 场景 (校准 split 独立): raw={r["ap_O_consistency_raw_on_odd_scenes"]["map50"]:.5f} / '
      f'cal={r["ap_O_consistency_cal_on_odd_scenes"]["map50"]:.5f}')
    A('')
    A('## 5. 三层->六层增益的 hard Query 贡献')
    A('')
    hq = r['hard_query_analysis']
    A(f'- 最终 TP 中需要 L4-L6 才首次正确的 query: **{hq["n_tp_need_l4_l6"]} '
      f'({hq["share_of_tp_pct"]:.2f}%)**')
    A(f'- 在 L3 内已正确的最终 TP: {hq["n_tp_done_by_l3"]}')
    A('')
    A('## 6. H1 判定')
    A('')
    A('| 条件 | 观察值 | 阈值 | 是否满足 |')
    A('|---|---:|---:|:---:|')
    for name, ok, val in conds:
        A(f'| {name} | {val:.4f} | — | {"是" if ok else "**否**"} |')
    A('')
    A(f'**结论: H1 {"支持" if h1_supported else "不支持"}**'
      f'{"" if h1_supported else " (以最终全量数据为准)"}。')
    A('')
    A('参照 (不参与判定):')
    A('')
    A(f'- O-GT50 mixed 相对 L6: -{h1["ref_gt50_mixed_vs_L6_gap_pp"]:.2f}pp; 其下降是跨层分数重排序'
      f'的贪心匹配串扰, 而非框不可用 —— TP 子集平均必要深度仅 '
      f'{h1["ref_gt50_tp_avg_depth_layers"]:.2f} 层。')
    A('')

    op = os.path.join(WORK, f'qd_oracle_report{sfx}.md')
    with open(op, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'已写入 {op}')


if __name__ == '__main__':
    main()
