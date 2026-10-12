# -*- coding: utf-8 -*-
"""
verify_ap_vs_pycoco.py —— 用 pycocotools 校验自实现的 AP
========================================================
为什么不用 ultralytics 当基准:
    ultralytics 的 ap_per_class 用 1000 点 + np.trapezoid 积分, 且精度曲线
    前置 1.0 再取后缀最大, 与 COCO 官方的 101 点采样不同 —— 它是近似实现。
    而流式感知文献(含官方 Argoverse-HD 评测)用的是 COCO 口径。
    因此自实现的 AP 应当对齐 pycocotools, 而不是对齐 ultralytics。

做法:
    1. 取录制好的预测(npz), 导出为 COCO 结果格式
    2. 用 pycocotools.COCOeval 算 AP50 / AP
    3. 调用 eval_sap.evaluate(verify=True) 算同一批预测, 对比两者
       (verify 模式下关联规则退化为"对比自己", 即普通 AP 的语义)

注意 maxDets: COCO 默认 100, 但 RT-DETR 每帧输出 300 个查询。两边都要显式
设成同一个值(本脚本用 300), 否则对比不公平。

用法:
    python verify_ap_vs_pycoco.py --preds stream_preds_test/preds_d3.npz --scenes 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from eval_sap import (build_timeline, load_gt, evaluate, assoc_floor)  # noqa: E402

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
         'traffic_light', 'stop_sign']


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', required=True, help='录制好的 npz')
    ap.add_argument('--scenes', nargs='+', type=int, default=None,
                    help='只评这几个场景; 不给则用录制里的全部帧')
    ap.add_argument('--ann', default=os.path.join(DATA, 'annotations_val.json'))
    ap.add_argument('--stream-index', default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--max-det', type=int, default=300)
    args = ap.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    frames, scene_t, scene_files = build_timeline(args.stream_index)
    frame_key = {(f[0], f[1]): i for i, f in enumerate(frames)}
    z = np.load(args.preds, allow_pickle=False)
    rec = {k: z[k] for k in z.files}

    # 选择要评的帧(默认: 录制覆盖的全部)
    sel = np.arange(len(rec['scene']))
    if args.scenes is not None:
        sel = np.array([i for i, s in enumerate(rec['scene']) if int(s) in args.scenes])

    # --- 1. 导出 COCO 检测结果 ---
    results = []
    for i in sel:
        s = rec['ptr'][i]
        e = rec['ptr'][i + 1]
        if e <= s:
            continue
        fn = scene_files[int(rec['scene'][i])][int(rec['frame_idx'][i])]
        for b, sc, c in zip(rec['box'][s:e], rec['score'][s:e], rec['cls'][s:e]):
            x1, y1, x2, y2 = (float(v) for v in b)
            results.append(dict(image_id=fn, category_id=int(c) + 1,
                                bbox=[x1, y1, x2 - x1, y2 - y1], score=float(sc)))
    log(f'导出 {len(results)} 个检测结果, 覆盖 {len(sel)} 帧')

    # --- 2. 构造该子集的 GT(COCO 格式) ---
    with open(args.ann, encoding='utf-8') as f:
        coco_gt = json.load(f)
    keep_files = {scene_files[int(rec['scene'][i])][int(rec['frame_idx'][i])]
                  for i in sel}
    idmap = {im['file_name']: im['id'] for im in coco_gt['images']}
    keep_ids = {idmap[f] for f in keep_files if f in idmap}
    sub = dict(
        info=coco_gt.get('info', {}),
        images=[im for im in coco_gt['images'] if im['id'] in keep_ids],
        annotations=[a for a in coco_gt['annotations'] if a['image_id'] in keep_ids],
        categories=coco_gt['categories'])
    sub_path = '_tmp_scene_gt.json'
    with open(sub_path, 'w', encoding='utf-8') as f:
        json.dump(sub, f, ensure_ascii=False)
    log(f'子集 GT: {len(sub["images"])} 图 / {len(sub["annotations"])} 框')

    # 结果里的 image_id 用文件名, 先翻译成数字 id
    for r in results:
        r['image_id'] = idmap[r['image_id']]

    coco = COCO(sub_path)
    if not results:
        log('[错误] 没有检测结果')
        return
    dt = coco.loadRes(results)
    # maxDets 只用 300: summarize() 的 stats 会为每个 maxDets 存一段,
    # 若列表里含 100, stats[0] 就是 maxDets=100 那段(可能全 -1), 容易读错。
    ev = COCOeval(coco, dt, iouType='bbox')
    ev.params.maxDets = [args.max_det]
    ev.evaluate()
    ev.accumulate()
    # precision: [T,R,K,A,M]; 取 area=all(index 0)、maxDets 唯一那段(index 0)
    prec = ev.eval['precision'][:, :, :, 0, 0]
    valid = prec[prec > -1]
    coco_ap = float(valid.mean()) if valid.size else 0.0
    coco_ap50 = float(prec[0][prec[0] > -1].mean()) if (prec[0] > -1).any() else 0.0
    log(f'pycocotools: AP50={coco_ap50:.4f}  AP50-95={coco_ap:.4f}  '
        f'(IoU 档数 {len(ev.params.iouThrs)}, maxDets={args.max_det})')

    # --- 3. 自实现(verify 模式 = 对比自己, 即普通 AP) ---
    gt = load_gt(args.ann)
    # 只保留子集帧, 保证分母一致
    gt_sub = {f: gt[f] for f in keep_files}
    my = evaluate(rec, gt_sub, frames, scene_t, scene_files, 0.0,
                  verify=True, frame_key=frame_key)
    log('')
    log('=' * 62)
    log(f"{'指标':<16}{'pycocotools':>16}{'自实现':>12}{'差值':>10}")
    log(f"{'AP50':<16}{coco_ap50:>16.4f}{my['sap50']:>12.4f}"
        f"{my['sap50']-coco_ap50:>+10.4f}")
    log(f"{'AP50-95':<16}{coco_ap:>16.4f}{my['sap']:>12.4f}"
        f"{my['sap']-coco_ap:>+10.4f}")
    log('=' * 62)
    log('差值应在百分位小数量级(±0.005 内可认为实现正确)')
    try:
        os.remove(sub_path)
    except OSError:
        pass


if __name__ == '__main__':
    main()
