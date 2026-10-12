# -*- coding: utf-8 -*-
"""
make_train_subset.py —— 生成"抽帧训练"子集 (方案一: 本地可行性验证)
=====================================================================
背景与动机:
    Argoverse-HD train 全量 39384 张, 平均每图 19.6 个目标(COCO 仅 7.3),
    RT-DETR 的匈牙利匹配与 decoder loss 按目标数计费, 单样本成本是 COCO 的
    2.7 倍。实测本机 5070 Ti-Laptop(功耗锁 140W, SM 仅跑 907/3090 MHz) 在
    imgsz=640/batch=8 下只有 1.9 it/s, 即 43 分钟/epoch:
        全量 200 epoch -> 143 小时 (不可行)
        全量  50 epoch ->  36 小时
    本脚本改为抽帧训练: 数据量降到 1/5, 换取足够 epoch 数。
    Argoverse-HD 是 30 FPS 连续视频, 相邻帧高度冗余, 抽帧对精度的损害
    远小于"把 epoch 从 100 砍到 20"。

与冒烟子集的区别(为什么不用 make_smoke_subset.py):
    冒烟版是纯随机抽样, 可能把某个场景抽空; 本脚本按 scene 分层抽样,
    保证 21 个场景都有代表 —— 这对后续按时序做分析是必要的。

用法:
    # 抽 20% (约 7870 张)
    python make_train_subset.py --ratio 0.2

    # 指定张数
    python make_train_subset.py --n 8000

    # val 保持全量(验证集不能抽, 否则指标不可比)
    python make_train_subset.py --ratio 0.2 --val-ratio 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic_light', 'stop_sign']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=r'D:\lunwen\自研\数据集')
    ap.add_argument('--yolo-dir', default='argoverse-hd-yolo', help='YOLO 布局目录名')
    ap.add_argument('--stream-index', default=None,
                    help='时序索引 json; 默认优先取 stream_index_train.json,'
                         ' 回退到 stream_index.json')
    ap.add_argument('--ratio', type=float, default=0.2, help='抽取比例(按 scene 分层)')
    ap.add_argument('--n', type=int, default=0, help='直接指定张数(优先于 --ratio)')
    ap.add_argument('--val-ratio', type=float, default=1.0, help='val 抽取比例(默认全量)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--name', default='train_sub', help='输出清单前缀')
    args = ap.parse_args()

    root = os.path.join(args.data_root, args.yolo_dir)
    if args.stream_index:
        idx_path = args.stream_index
    else:
        idx_dir = os.path.join(args.data_root, 'argoverse-hd')
        for cand in ('stream_index_train.json', 'stream_index.json'):
            if os.path.exists(os.path.join(idx_dir, cand)):
                idx_path = os.path.join(idx_dir, cand)
                break
        else:
            print(f'[错误] 在 {idx_dir} 下找不到 stream_index_train.json 或 stream_index.json')
            print('       请先跑: python build_stream_index.py --split train ...')
            sys.exit(1)

    if not os.path.exists(idx_path):
        print(f'[错误] 找不到时序索引: {idx_path}')
        sys.exit(1)

    with open(idx_path, encoding='utf-8') as f:
        idx = json.load(f)
    scenes = idx['scenes']
    print(f'读取 {idx_path}')
    print(f'  {idx["n_frames"]} 帧 / {idx["n_scenes"]} 场景, 相机 {idx["camera"]}')

    rng = random.Random(args.seed)
    img_dir = os.path.join(root, 'images', 'train')

    # 只保留实际存在且标签非空的帧
    def usable(fn):
        ip = os.path.join(img_dir, fn)
        lp = os.path.join(root, 'labels', 'train', os.path.splitext(fn)[0] + '.txt')
        return os.path.exists(ip) and os.path.exists(lp) and os.path.getsize(lp) > 0

    # 按目标比例反推每场景应抽多少(分层), 保证长场景多抽、短场景少抽
    per_scene = []
    total_avail = 0
    for s in scenes:
        avail = [fr['file'] for fr in s['frames'] if usable(fr['file'])]
        per_scene.append((s['scene_id'], avail))
        total_avail += len(avail)

    if total_avail == 0:
        print('[错误] 没有可用帧(图片或标签缺失)。先跑 build_yolo_dataset.py')
        sys.exit(1)

    target = args.n if args.n > 0 else int(round(total_avail * args.ratio))
    target = min(target, total_avail)
    print(f'')
    print(f'训练集可用 {total_avail} 帧, 目标抽取 {target} 张 ({target/total_avail*100:.1f}%)')

    picked, shortfall = [], 0
    for sid, avail in per_scene:
        if not avail:
            continue
        want = int(round(len(avail) * target / total_avail))
        want = min(want, len(avail))
        if want == 0 and len(avail) > 0:
            want = 1                      # 短场景至少留 1 帧, 避免整段丢失
        take = rng.sample(avail, want)
        picked.extend((sid, f) for f in take)
        if want < int(round(len(avail) * target / total_avail)):
            shortfall += 1

    print(f'分层结果: {len(per_scene)} 个场景全部覆盖')
    # 打印每个场景的抽取情况, 便于人工核对
    by_scene = {}
    for sid, f in picked:
        by_scene.setdefault(sid, 0)
        by_scene[sid] += 1
    for s in scenes:
        sid = s['scene_id']
        print(f'  scene {sid:>3}: 抽 {by_scene.get(sid,0):>5} / {s["n_frames"]:>5} 帧')

    out_txt = os.path.join(root, f'{args.name}.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        for sid, fn in picked:
            # 必须写绝对路径: ultralytics 的 img2label_paths() 用
            #   x.rsplit("\images\", 1)
            # 把 images 段替换成 labels 段, 这要求 "\images\" 之前有前导分隔符。
            # 写成相对路径 "images/train/a.jpg" 会导致替换失败, 报
            # "No labels found in images\train.cache"。
            f.write(os.path.join(img_dir, fn) + '\n')
    print(f'')
    print(f'训练清单 -> {out_txt}  ({len(picked)} 行)')

    # val
    val_dir = os.path.join(root, 'images', 'val')
    lbl_dir = os.path.join(root, 'labels', 'val')
    if os.path.isdir(val_dir):
        vall = sorted(f for f in os.listdir(val_dir) if f.lower().endswith('.jpg'))
        if args.val_ratio < 1.0:
            vall = rng.sample(vall, int(len(vall) * args.val_ratio))
            print(f'val 抽 {args.val_ratio*100:.0f}% -> {len(vall)} 张')
        else:
            print(f'val 保持全量 -> {len(vall)} 张')
        with open(os.path.join(root, f'{args.name}_val.txt'), 'w', encoding='utf-8') as f:
            for fn in vall:
                f.write(os.path.join(val_dir, fn) + '\n')

    yaml_path = os.path.join(root, f'{args.name}.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f'# 自动生成 by make_train_subset.py (方案一: 抽帧训练)\n')
        f.write(f'# 训练集 {len(picked)} 张 (全量 {total_avail} 的 {len(picked)/total_avail*100:.1f}%)\n')
        f.write(f'path: {root}\n')
        f.write(f'train: {args.name}.txt\n')
        f.write(f'val: {args.name}_val.txt\n')
        f.write('names:\n')
        for i, n in enumerate(NAMES):
            f.write(f'  {i}: {n}\n')
    print(f'data yaml -> {yaml_path}')


if __name__ == '__main__':
    main()
