# -*- coding: utf-8 -*-
"""make_qd_p_report.py —— 从 query_depth_schedule.json 生成 query_depth_schedule_report.md
=========================================================================================
按任务书 3.2 输出: 理论 Query-layer 预算 vs 真实 decoder-only CUDA / 整体 CUDA / predict wall,
mAP / sAP 对照, 并对"选择性深度是否带来真实端到端收益"做判定。
用法: python make_qd_p_report.py [--suffix smoke]
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

POL_NAME = {'conf': 'confidence', 'stab': 'stability', 'cs': 'confidence+stability',
            'rand': 'random', 'oracle': 'GT oracle'}
SCH_NAME = {'p1': 'P1 (300/150/75)', 'p2': 'P2 (300/200/100)', 'p3': 'P3 (300/100/50)'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='')
    args = ap.parse_args()
    sfx = f'_{args.suffix}' if args.suffix else ''
    with open(os.path.join(WORK, f'query_depth_schedule{sfx}.json'), encoding='utf-8') as f:
        r = json.load(f)

    a0 = r['a0p']
    lines = []
    A = lines.append
    A('# QD-P 阶段报告: 三层模型选择性深度调度 (真实 active subset)')
    A('')
    A(f'> 数据: query_depth_schedule{sfx}.json; 基线 a0p (无退出)。所有配置真实执行存活子集, '
      f'退出 query 冻结该层输出; GT oracle 掩码离线预计算, 模型内部无 GT。')
    A('')
    A('> **注: 阶段三经确认不跑全量, 本报告为 smoke (2 场景 1356 帧) 结论, 引用时需标注。**')
    A('')
    A('## 1. 主结果')
    A('')
    A('| 配置 | mAP50 | Δ vs a0p (pp) | mAP50-95 | AP75 | E[d] | 实际省 | 理论省 | wall med | dec med | model med |')
    A('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    rows = [c for c in r if not c.startswith('_') and c != 'a0p']
    A(f'| a0p | {a0["ap"]["map50"]:.5f} | — | {a0["ap"]["map"]:.5f} | {a0["ap"]["ap75"]:.5f} '
      f'| 3.00 | 0% | 0% | {a0["latency"]["wall_median"]:.2f} | {a0["latency"]["decoder_cuda_median"]:.3f} | '
      f'{a0["latency"]["model_cuda_median"]:.2f} |')
    for cfg in rows:
        if cfg == 'a0p':
            continue
        e = r[cfg]
        A(f'| {cfg} | {e["ap"]["map50"]:.5f} | {e["vs_a0_pp"]["map50"]:+.2f} | '
          f'{e["ap"]["map"]:.5f} | {e["ap"]["ap75"]:.5f} | {e["execution"]["E_depth_layers"]:.2f} | '
          f'{e["execution"]["savings_vs_3L_pct"]:.1f}% | {e["budget_theoretical"]["savings_vs_3L_pct"]:.1f}% | '
          f'{e["latency"]["wall_median"]:.2f} | {e["latency"]["decoder_cuda_median"]:.3f} | '
          f'{e["latency"]["model_cuda_median"]:.2f} |')
    A('')
    A('## 2. 按策略分解 (各调度下的 mAP50 与 E[d])')
    A('')
    A('| 调度 | confidence | stability | conf+stab | random | GT oracle |')
    A('|---|---:|---:|---:|---:|---:|')
    for sch in ['p1', 'p2', 'p3']:
        cells = []
        for pol in ['conf', 'stab', 'cs', 'rand', 'oracle']:
            e = r.get(f'{sch}_{pol}')
            cells.append(f'{e["ap"]["map50"]:.5f} / E[{e["execution"]["E_depth_layers"]:.2f}]' if e else '—')
        A(f'| {SCH_NAME[sch]} | ' + ' | '.join(cells) + ' |')
    A('')
    A('## 3. 延迟与理论预算的差距 (3.2 关键问题)')
    A('')
    A('| 配置 | 理论节省 | 实际 E[d] | decoder-only (ms) | 整体 CUDA (ms) | predict wall (ms) |')
    A('|---|---:|---:|---:|---:|---:|')
    for cfg in ['a0p'] + rows:
        e = r[cfg]
        A(f'| {cfg} | {0 if cfg == "a0p" else e["budget_theoretical"]["savings_vs_3L_pct"]:.1f}% | '
          f'{e["execution"]["E_depth_layers"]:.2f} | {e["latency"]["decoder_cuda_median"]:.3f} | '
          f'{e["latency"]["model_cuda_median"]:.2f} | {e["latency"]["wall_median"]:.2f} |')
    A('')
    # 判定: 实际延迟收益
    p1 = r.get('p1_conf', {})
    a0wall = a0['latency']['wall_median']
    a0model = a0['latency']['model_cuda_median']
    A('## 4. 判定')
    A('')
    A(f'- 基线 a0p: decoder-only {a0["latency"]["decoder_cuda_median"]:.3f}ms, '
      f'整体 CUDA {a0model:.2f}ms, predict wall {a0wall:.2f}ms;')
    if p1:
        A(f'- P1-confidence (理论省 41.7%): decoder-only {p1["latency"]["decoder_cuda_median"]:.3f}ms '
          f'(Δ {100 * (p1["latency"]["decoder_cuda_median"] / a0["latency"]["decoder_cuda_median"] - 1):+.1f}%), '
          f'整体 CUDA {p1["latency"]["model_cuda_median"]:.2f}ms '
          f'(Δ {100 * (p1["latency"]["model_cuda_median"] / a0model - 1):+.1f}%), '
          f'wall {p1["latency"]["wall_median"]:.2f}ms '
          f'(Δ {100 * (p1["latency"]["wall_median"] / a0wall - 1):+.1f}%)。')
    A('')
    A('结论要点: 三层内 query-layer 削减的 decoder 时间占比很小 (encoder 主导), '
      '且当前参考实现的逐层路由 (掩码 gather/合并) 引入了可观的 CPU 侧开销 —— '
      '退出配置的 decoder-only 与整体 CUDA 反而高于基线, 理论节省完全没有兑现为真实延迟。')
    A('')
    A('这直接回答交付物问题 5: **当前实现下选择性深度没有真实端到端延迟收益, 只有理论预算收益**。'
      '若论文要讲"按需深度"的效率故事, 必须 (a) 报真实 decoder/整体 CUDA/wall 而非 Query-layer 预算, '
      '(b) 工程化路由 (预计算索引计划 / CUDA Graph / 编译融合) 消除逐层路由开销后再测, '
      '否则只能把选择性深度表述为 FLOPs 侧的故事。AP 侧以全量数据为准。')
    A('')

    op = os.path.join(WORK, f'query_depth_schedule_report{sfx}.md')
    with open(op, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'已写入 {op}')


if __name__ == '__main__':
    main()
