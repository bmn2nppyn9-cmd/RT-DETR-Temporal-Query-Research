# -*- coding: utf-8 -*-
"""
build_yolo_dataset.py —— Argoverse-HD COCO 标注 -> ultralytics YOLO 训练布局
============================================================================
为什么需要这个脚本:
    prepare_argoverse.py 产出的是 COCO json (images/annotations/categories),
    但 ultralytics 训练读的是 YOLO txt 标签, 且目录布局要求:
        <root>/
          images/{train,val}/*.jpg
          labels/{train,val}/*.txt
    另外类目 id 也要对齐: 本地 COCO 是 category_id = 1..8, 而训练需要 0..7。

三个必须小心的点:
  1. 类目 id 要减 1 (COCO 的 1-based -> YOLO 的 0-based), 否则 8 会越界
     (nc=8 时合法 id 是 0-7, 出现 8 会被 ultralytics 直接丢弃并告警)
  2. 归一化要用每张图自己的 width/height, 不能写死 1920/1200
  3. 图片默认用符号链接硬指回原平铺目录, 不复制。20 GB 不想存两份。
     若系统不支持软链(需管理员或开发者模式), 加 --copy 回退。

用法:
    # 默认: 建 train/val 两个 split 的软链布局 + txt 标签
    python build_yolo_dataset.py

    # 只建 val (调试用, 秒级)
    python build_yolo_dataset.py --splits val

    # 不支持软链时的回退 (会真实复制图片, 占额外磁盘)
    python build_yolo_dataset.py --copy
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def log(m):
    print(m, flush=True)


def link_or_copy(src, dst, copy_mode):
    """把 src 以软链(或复制)方式落到 dst; 已存在则跳过"""
    if os.path.exists(dst) or os.path.islink(dst):
        return 'skip'
    try:
        if copy_mode:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)
        return 'ok'
    except OSError as e:
        # Windows 上 symlink 需要权限; 回退到硬链接, 再不行就复制
        try:
            os.link(src, dst)
            return 'hardlink'
        except OSError:
            shutil.copy2(src, dst)
            return 'copy'


def convert_split(split, img_src_dir, root, cat_shift, copy_mode, ann_dir=None):
    """转换一个 split, 返回统计 dict

    ann_dir: annotations_<split>.json 所在目录; 默认取 root 的上一级。
             本地布局里 root=数据集/argoverse-hd-yolo, 而标注在 数据集/argoverse-hd/,
             两者不同级, 所以必须显式传, 不能只靠 dirname。
    """
    ann_path = os.path.join(ann_dir or os.path.dirname(root), f'annotations_{split}.json')
    if not os.path.exists(ann_path):
        log(f'  [跳过] 找不到 {ann_path}')
        return None

    log(f'  读取 {ann_path}')
    with open(ann_path, encoding='utf-8') as f:
        coco = json.load(f)

    img_out = os.path.join(root, 'images', split)
    lbl_out = os.path.join(root, 'labels', split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    # image_id -> (file_name, width, height)
    id2img = {im['id']: im for im in coco['images']}
    # image_id -> [标注行]
    per_img = {im['id']: [] for im in coco['images']}

    n_skip_cls = 0
    cls_seen = Counter()
    for a in coco['annotations']:
        iid = a['image_id']
        im = id2img.get(iid)
        if im is None:
            continue
        cid = int(a['category_id']) + cat_shift      # COCO 1-based -> YOLO 0-based
        if cid < 0:
            n_skip_cls += 1
            continue
        cls_seen[cid] += 1
        W, H = float(im['width']), float(im['height'])
        x, y, w, h = (float(v) for v in a['bbox'])
        # COCO 是左上角 xywh; YOLO 要中心点归一化 xywh
        cx = (x + w / 2.0) / W
        cy = (y + h / 2.0) / H
        nw = w / W
        nh = h / H
        # 裁剪到 [0,1]: 原始标注有越界框, 不裁会让 loss 发散
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        nw, nh = min(max(nw, 0.0), 1.0), min(max(nh, 0.0), 1.0)
        if nw <= 0 or nh <= 0:
            continue
        per_img[iid].append(f'{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n')

    # 落盘: 图片软链 + 标签 txt
    n_img = n_lbl = n_empty = 0
    link_stat = Counter()
    for im in coco['images']:
        fn = im['file_name']
        stem = os.path.splitext(fn)[0]
        src = os.path.join(img_src_dir, fn)
        if not os.path.exists(src):
            continue                                  # 还没下载完
        link_stat[link_or_copy(src, os.path.join(img_out, fn), copy_mode)] += 1
        n_img += 1
        lines = per_img[im['id']]
        # 空标注也写一个空 txt: ultralytics 要求每张图都有对应标签文件,
        # 否则会把它当背景图并反复告警
        with open(os.path.join(lbl_out, stem + '.txt'), 'w', encoding='utf-8') as f:
            f.writelines(lines)
        n_lbl += 1
        if not lines:
            n_empty += 1

    log(f'  图片 -> {img_out}  ({n_img} 张, {dict(link_stat)})')
    log(f'  标签 -> {lbl_out}  ({n_lbl} 个, 其中空标签 {n_empty})')
    log(f'  类别框数 ' + ', '.join(f'{k}:{v}' for k, v in sorted(cls_seen.items())))
    if n_skip_cls:
        log(f'  [警告] {n_skip_cls} 个标注因类别越界被跳过')

    return dict(split=split, n_img=n_img, n_lbl=n_lbl, n_empty=n_empty,
                cls_seen=dict(cls_seen), link_stat=dict(link_stat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=r'D:\lunwen\自研\数据集',
                    help='存放 argoverse-hd 与 annotations_*.json 的目录')
    ap.add_argument('--src-name', default='argoverse-hd', help='原始平铺图片所在子目录名')
    ap.add_argument('--out-name', default='argoverse-hd-yolo', help='输出 YOLO 布局目录名')
    ap.add_argument('--splits', nargs='+', default=['train', 'val'],
                    help='要转换的 split (test 无标注, 不要转)')
    ap.add_argument('--cat-shift', type=int, default=-1,
                    help='COCO category_id -> YOLO id 的偏移 (默认 -1, 即 1..8 -> 0..7)')
    ap.add_argument('--copy', action='store_true', help='复制图片而非软链')
    ap.add_argument('--check-only', action='store_true', help='只检查原图是否齐备, 不建布局')
    args = ap.parse_args()

    img_src_dir = os.path.join(args.data_root, args.src_name, 'images')
    ann_dir = os.path.join(args.data_root, args.src_name)   # annotations_*.json 在这里
    root = os.path.join(args.data_root, args.out_name)

    if not os.path.isdir(img_src_dir):
        log(f'[错误] 原图目录不存在: {img_src_dir}')
        sys.exit(1)

    n_have = len([f for f in os.listdir(img_src_dir) if f.lower().endswith('.jpg')])
    log(f'[0/2] 原图目录现有 {n_have} 张 jpg')

    if args.check_only:
        return

    log(f'[1/2] 建立 YOLO 布局 -> {root}')
    log(f'      图片方式: {"复制" if args.copy else "软链(省磁盘)"}')
    results = []
    for split in args.splits:
        log(f'--- split = {split} ---')
        r = convert_split(split, img_src_dir, root, args.cat_shift, args.copy, ann_dir)
        if r:
            results.append(r)

    log('[2/2] 汇总')
    for r in results:
        log(f'  {r["split"]:<6} 图片 {r["n_img"]:>6}  标签 {r["n_lbl"]:>6}  空 {r["n_empty"]}')

    # 提示: 训练时若某些图还没下载完, 需要重跑本脚本补齐(已存在的会 skip)
    log('')
    log('注意: 若下载尚未完成, 现在转换会漏掉未下完的图。')
    log('      下载完成后需再跑一次本脚本补齐(已存在的软链会 skip)。')


if __name__ == '__main__':
    main()
