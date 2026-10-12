# -*- coding: utf-8 -*-
"""
make_smoke_subset.py —— 为训练冒烟测试生成小子集清单
=====================================================
目的: 在跑 8-10 小时的完整训练前, 先用几百张图跑通 2 个 epoch, 验证:
  - 数据管道能读到图片与标签(类目 id 不越界)
  - 损失能下降、不出现 NaN (RT-DETR 的 AMP 已知会 NaN)
  - 显存占用在 12.8GB 以内
  - 最后能正常出 val 指标与权重文件

ultralytics 的 train/val 字段支持 .txt 清单文件, 每行一个图片路径。

用法:
    python make_smoke_subset.py --n 400
"""
from __future__ import annotations

import argparse
import os
import random
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'D:\lunwen\自研\数据集\argoverse-hd-yolo')
    ap.add_argument('--n', type=int, default=400, help='子集图片数')
    ap.add_argument('--val-n', type=int, default=100, help='子集验证图片数')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    def dump(split, n, out_name):
        img_dir = os.path.join(args.root, 'images', split)
        if not os.path.isdir(img_dir):
            print(f'[跳过] 不存在 {img_dir}')
            return None
        imgs = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        if not imgs:
            print(f'[跳过] {img_dir} 为空')
            return None
        # 只取有对应标签文件的(理论上都有), 并优先取非空标签的图,
        # 避免冒烟阶段全是背景图看不出 loss 是否在学
        lbl_dir = os.path.join(args.root, 'labels', split)
        good = []
        for f in imgs:
            lp = os.path.join(lbl_dir, os.path.splitext(f)[0] + '.txt')
            if os.path.exists(lp) and os.path.getsize(lp) > 0:
                good.append(f)
        pool = good if len(good) >= n else imgs
        picked = rng.sample(pool, min(n, len(pool)))
        out = os.path.join(args.root, out_name)
        with open(out, 'w', encoding='utf-8') as fh:
            for f in picked:
                fh.write(os.path.join(img_dir, f) + '\n')
        print(f'{split:<6} 候选 {len(pool):>6} (有标签 {len(good)}) -> 取 {len(picked):>4}  '
              f'写入 {out}')
        return out

    print(f'数据集根目录: {args.root}')
    tr = dump('train', args.n, 'smoke_train.txt')
    va = dump('val', args.val_n, 'smoke_val.txt')

    if tr and va:
        print('')
        print('冒烟测试用 data yaml (yaml 里的 train/val 指向这两个清单):')
        print(f'  已生成 -> {os.path.join(args.root, "smoke.yaml")}')
        with open(os.path.join(args.root, 'smoke.yaml'), 'w', encoding='utf-8') as fh:
            fh.write('# 自动生成: 训练冒烟测试子集, 不要用于正式训练\n')
            fh.write(f'path: {args.root}\n')
            fh.write('train: smoke_train.txt\n')
            fh.write('val: smoke_val.txt\n')
            fh.write('names:\n')
            for i, n in enumerate(['person', 'bicycle', 'car', 'motorcycle',
                                   'bus', 'truck', 'traffic_light', 'stop_sign']):
                fh.write(f'  {i}: {n}\n')


if __name__ == '__main__':
    main()
