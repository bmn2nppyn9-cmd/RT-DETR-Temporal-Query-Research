# -*- coding: utf-8 -*-
"""
merge_3d.py —— 独立重合并: timing_3d.json + scan_3d_02.json + full_3d.json -> CSV
================================================================================
与 eval_grid_3d.py 的 merge 阶段同逻辑, 供手工改完 JSON 后重出 CSV。
优先取 full(含二维导入), 否则取 scan。不用 pandas, 纯 csv 模块。

用法:  python merge_3d.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEPTHS = [1, 3, 6]
QUERIES = [150, 300, 375]
RESOLUTIONS = [480, 512, 640]

timing = json.load(open(os.path.join(HERE, 'timing_3d.json'), encoding='utf-8'))
scan = json.load(open(os.path.join(HERE, 'scan_3d_02.json'), encoding='utf-8'))
full = json.load(open(os.path.join(HERE, 'full_3d.json'), encoding='utf-8'))

rows = []
for d in DEPTHS:
    for q in QUERIES:
        for r in RESOLUTIONS:
            k = f'd{d}_q{q}_r{r}'
            row = timing.get(k, {}).copy()
            acc = full.get(k, scan.get(k, {}))
            row.update(acc)          # full 优先于 scan; 无 acc 则留空
            rows.append(row)

cols = ['depth', 'queries', 'resolution', 'map50', 'map50_95',
        'median_ms', 'p95_ms', 'p99_ms', 'fps', 'gflops',
        'fraction', 'params_M', 'source']
out = os.path.join(HERE, 'ablation_3d_results.csv')
with open(out, 'w', encoding='utf-8', newline='') as f:
    f.write(','.join(cols) + '\n')
    for row in rows:
        f.write(','.join(str(row.get(c, '')) for c in cols) + '\n')
print(f'已生成: {out}  ({len(rows)} 行)')
