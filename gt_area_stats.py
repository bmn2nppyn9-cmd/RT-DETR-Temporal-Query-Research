# -*- coding: utf-8 -*-
"""
gt_area_stats.py —— Argoverse-HD val 集 GT 尺寸分布统计
=========================================================
目的: 决定论文中"按目标尺寸分档 AP"的分档策略。
    COCO 官方分档(面积 32^2 / 96^2 像素)是按 COCO 数据集的目标尺度定的,
    Argoverse-HD 是车载远距视角, 目标普遍很小, 官方分档可能退化
    (几乎全部目标落进同一档, 分档失去意义)。先用本脚本看真实分布,
    再决定用 COCO 档还是自定义像素档。

输出:
    打印: 总体分布 + 逐类分布 + 两种分档口径下的占比
    文件: out_gt_area_stats.json

用法:
    python gt_area_stats.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
ANN = os.path.join(DATA, 'annotations_val.json')

# COCO 官方面积分档(像素^2): small < 32^2 <= medium < 96^2 <= large
COCO_BINS = [0.0, 32.0 ** 2, 96.0 ** 2, float('inf')]
COCO_NAMES = ['small(<32px)', 'medium(32-96)', 'large(>96px)']

# 候选自定义分档: 按框的 sqrt(area)(等效边长, 像素), 更贴合小目标数据集
CUSTOM_BINS = [0.0, 16.0, 32.0, 64.0, 96.0, float('inf')]
CUSTOM_NAMES = ['tiny(<16px)', 'small(16-32)', 'mid(32-64)',
                'large(64-96)', 'huge(>96px)']


def pct_table(areas, bins, names):
    frac, _ = np.histogram(areas, bins=bins)
    frac = frac / max(len(areas), 1)
    return {n: float(f) for n, f in zip(names, frac)}


def main():
    with open(ANN, encoding='utf-8') as f:
        coco = json.load(f)
    cats = [c['name'] for c in sorted(coco['categories'], key=lambda c: c['id'])]
    ann = coco['annotations']
    wh = np.asarray([[a['bbox'][2], a['bbox'][3]] for a in ann], np.float32)
    cid = np.asarray([a['category_id'] - 1 for a in ann], np.int64)
    area = wh[:, 0] * wh[:, 1]
    side = np.sqrt(area)

    print(f'GT 框总数: {len(area)}  图像: {len(coco["images"])}')
    print(f'面积(px^2)  中位 {np.median(area):.0f}  '
          f'P10 {np.percentile(area,10):.0f}  P90 {np.percentile(area,90):.0f}')
    print(f'等效边长(px) 中位 {np.median(side):.1f}  '
          f'P10 {np.percentile(side,10):.1f}  P90 {np.percentile(side,90):.1f}')
    print(f'宽(px) 中位 {np.median(wh[:,0]):.1f}   '
          f'高(px) 中位 {np.median(wh[:,1]):.1f}')
    print()

    out = {'n_gt': int(len(area)), 'n_images': len(coco['images']),
           'area_px2': {p: float(np.percentile(area, q)) for p, q in
                        [('P10', 10), ('P25', 25), ('P50', 50),
                         ('P75', 75), ('P90', 90), ('P99', 99)]},
           'side_px': {p: float(np.percentile(side, q)) for p, q in
                       [('P10', 10), ('P25', 25), ('P50', 50),
                        ('P75', 75), ('P90', 90), ('P99', 99)]},
           'coco_bins': pct_table(area, COCO_BINS, COCO_NAMES),
           'custom_bins': pct_table(side, CUSTOM_BINS, CUSTOM_NAMES),
           'per_class': {}}

    print('== COCO 官方分档占比 ==')
    for n, f in out['coco_bins'].items():
        print(f'  {n:<16}{f*100:6.2f}%')
    print()
    print('== 自定义等效边长分档占比 ==')
    for n, f in out['custom_bins'].items():
        print(f'  {n:<16}{f*100:6.2f}%')
    print()

    hdr = f"{'类别':<16}{'框数':>8}{'边长中位':>9}"
    for n in COCO_NAMES:
        hdr += f'{n:>14}'
    print(hdr)
    for c, name in enumerate(cats):
        m = cid == c
        a = area[m]
        row = pct_table(a, COCO_BINS, COCO_NAMES)
        out['per_class'][name] = {
            'n': int(m.sum()),
            'side_median': float(np.median(side[m])) if m.any() else 0.0,
            'coco_bins': row,
            'custom_bins': pct_table(side[m], CUSTOM_BINS, CUSTOM_NAMES)}
        line = f"{name:<16}{m.sum():>8}{np.median(side[m]):>9.1f}"
        for n in COCO_NAMES:
            line += f'{row[n]*100:>13.1f}%'
        print(line)

    with open('out_gt_area_stats.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print()
    print('已写入 out_gt_area_stats.json')


if __name__ == '__main__':
    main()
