# -*- coding: utf-8 -*-
"""
Argoverse-HD (Ultralytics ndjson 导出格式) -> 本地图片 + COCO 标注
=================================================================
输入: argoverse-hd.ndjson   每行一个 JSON 对象
    {"type":"dataset", ...}
    {"type":"image", "file":..., "url":..., "width":..., "height":...,
     "split":"val", "annotations": {"boxes": [[cls, cx, cy, w, h], ...]}}

输出:
    <out>/images/<file>            下载的图片
    <out>/annotations_<split>.json COCO 格式标注(供 exp_ab_adaptive_depth.py 使用)

用法:
  # 先下 30 张 val 打通流程
  python prepare_argoverse.py --split val --limit 30

  # 全量 val (~15062 张, 约 6.4GB)
  python prepare_argoverse.py --split val

  # 只要标注不要图片(快速生成 COCO json)
  python prepare_argoverse.py --split val --no-images
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle',
               'bus', 'truck', 'traffic_light', 'stop_sign']


def log(m):
    print(m, flush=True)


def iter_images(ndjson, split):
    """流式读取, 只产出指定 split 且带标注的图片条目"""
    with open(ndjson, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get('type') != 'image':
                continue
            if split != 'all' and o.get('split') != split:
                continue
            yield o


def download_one(item, img_dir, retries=3, timeout=60):
    """下载单张图片, 返回 (file, ok, size或错误信息)"""
    fn = item['file']
    dst = os.path.join(img_dir, fn)
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return fn, True, os.path.getsize(dst)   # 断点续传: 已存在则跳过
    for attempt in range(retries):
        try:
            req = Request(item['url'], headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=timeout) as r, open(dst, 'wb') as w:
                w.write(r.read())
            return fn, True, os.path.getsize(dst)
        except Exception as e:
            if attempt == retries - 1:
                return fn, False, f'{type(e).__name__}: {e}'
            time.sleep(1.5 * (attempt + 1))
    return fn, False, 'unknown'


def to_coco(items, out_path):
    """YOLO 归一化 xywh -> COCO xywh(像素), 同时写 images/annotations/categories"""
    images, annotations = [], []
    ann_id = 1
    for iid, it in enumerate(items, start=1):
        w, h = int(it['width']), int(it['height'])
        images.append(dict(id=iid, file_name=it['file'], width=w, height=h))
        boxes = (it.get('annotations') or {}).get('boxes') or []
        for b in boxes:
            try:
                cls, cx, cy, bw, bh = b[0], b[1], b[2], b[3], b[4]
            except Exception:
                continue
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            annotations.append(dict(
                id=ann_id, image_id=iid, category_id=int(cls) + 1,
                bbox=[round(x, 2), round(y, 2), round(bw * w, 2), round(bh * h, 2)],
                area=round(bw * w * bh * h, 2), iscrowd=0))
            ann_id += 1
    coco = dict(
        info=dict(description='Argoverse-HD (converted from ultralytics ndjson)'),
        images=images, annotations=annotations,
        categories=[dict(id=i + 1, name=n) for i, n in enumerate(CLASS_NAMES)])
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False)
    return coco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ndjson', default=r'D:\lunwen\自研\数据集\argoverse-hd.ndjson')
    ap.add_argument('--out', default=r'D:\lunwen\自研\数据集\argoverse-hd')
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test', 'all'])
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 张(0=全部)')
    ap.add_argument('--workers', type=int, default=16, help='并发下载线程数')
    ap.add_argument('--no-images', action='store_true', help='只生成标注, 不下载图片')
    ap.add_argument('--shard', type=int, default=0,
                    help='分片序号(0起)。与 --num-shards 配合, 可开多个进程分摊下载')
    ap.add_argument('--num-shards', type=int, default=1,
                    help='分片总数。单进程下载受"每连接延迟"限制(实测 6 线程仅 1.4 张/秒),'
                         '开 4 个进程各 15 线程可线性提速')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    img_dir = os.path.join(args.out, 'images')
    os.makedirs(img_dir, exist_ok=True)

    log(f'[1/3] 读取 {args.ndjson} (split={args.split})')
    items = []
    for o in iter_images(args.ndjson, args.split):
        items.append(o)
        if args.limit and len(items) >= args.limit:
            break

    # 分片: 按序号取模, 各进程负责不相交的子集(断点续传保证不冲突)
    if args.num_shards > 1:
        n_all = len(items)
        items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard]
        log(f'      分片 {args.shard}/{args.num_shards}: 本进程负责 {len(items)}/{n_all} 张')

    n_with_ann = sum(1 for it in items if (it.get('annotations') or {}).get('boxes'))
    log(f'      选中 {len(items)} 张, 其中 {n_with_ann} 张有标注')

    # 下载图片
    if not args.no_images:
        log(f'[2/3] 下载图片 -> {img_dir}  (并发 {args.workers})')
        t0 = time.time()
        done = ok = 0
        fails = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(download_one, it, img_dir) for it in items]
            for fu in as_completed(futs):
                fn, good, info = fu.result()
                done += 1
                if good:
                    ok += 1
                else:
                    fails.append((fn, info))
                if done % max(1, len(items) // 10) == 0 or done == len(items):
                    el = time.time() - t0
                    sp = done / el if el > 0 else 0
                    log(f'      {done}/{len(items)}  成功 {ok}  失败 {len(fails)}  '
                        f'{sp:.1f} 张/秒(分片口径)')
        log(f'      下载完成: 成功 {ok}, 失败 {len(fails)}, 用时 {(time.time()-t0)/60:.1f} 分钟')
        if fails:
            log(f'      失败示例: {fails[:3]}')
    else:
        log('[2/3] 跳过图片下载(--no-images)')

    # 生成 COCO 标注
    # 注意: 分片模式下不写标注, 否则每个分片都会用自己那部分覆盖整个 json
    if args.num_shards > 1:
        log('[3/3] 分片模式: 跳过标注生成(由单独一次 --no-images 全量生成)')
        return

    log('[3/3] 生成 COCO 标注')
    ann_path = os.path.join(args.out, f'annotations_{args.split}.json')
    coco = to_coco(items, ann_path)
    log(f'      已写入 {ann_path}')
    log(f'      images={len(coco["images"])}  annotations={len(coco["annotations"])}')

    log('\n下一步:')
    log(f'  python exp_ab_adaptive_depth.py infer \\')
    log(f'      --images "{img_dir}" \\')
    log(f'      --ann "{ann_path}" \\')
    log(f'      --weights <你的RT-DETR权重.pt> \\')
    log(f'      --out out_ab')


if __name__ == '__main__':
    main()
