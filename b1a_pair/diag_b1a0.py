# -*- coding: utf-8 -*-
"""
diag_b1a0.py —— b1a0 训练结果的独立 A/B 判别
==============================================
ultralytics validator 报 b1a0 mAP50=0.227, 与 3L 冻结基线 0.3865 严重不符。
本脚本绕开 validator, 用已验证的评测链(eval_sap 机制)在 val 前 400 帧上对比:
  A: 初始模型(baseline best.pt 加载进 ndl=3, 未训练)
  B: b1a0 训 1 epoch 的 best.pt
两者同帧同环境, 直接回答"训练是否真的把模型练坏"。
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, r'C:\Users\TONY\ultralytics')
sys.path.insert(0, r'D:\lunwen\自研')

import numpy as np
import torch

from eval_sap import build_timeline, greedy_tp, iou_matrix, load_gt, voc_ap
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.nn.tasks import RTDETRDetectionModel

HERE = os.path.dirname(os.path.abspath(__file__))
YAML = r'D:\lunwen\自研\cfg\rtdetr-resnet18.yaml'
CKPT_INIT = r'D:\lunwen\自研\runs\detect\runs_rtdetr\r18_640_sub20_b4\weights\best.pt'
CKPT_B1A0 = os.path.join(HERE, 'runs', 'b1a0', 'weights', 'best.pt')
DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
N_FRAMES = 400
NC = 8
IOU_THRS = np.round(np.arange(0.5, 1.0, 0.05), 2)


def build_and_load(ckpt_path, struct):
    RTDETRDecoder.b1a_struct_override = struct
    m = RTDETRDetectionModel(YAML, ch=3, nc=NC, verbose=False)
    RTDETRDecoder.b1a_struct_override = None
    m.load(torch.load(ckpt_path, map_location='cpu', weights_only=False))
    return m


def as_api(m, tag):
    """nn 模型 -> 高层 API(自带 letterbox/坐标还原), 经临时 ckpt 走标准加载链"""
    from ultralytics import RTDETR
    tmp = os.path.join(HERE, 'logs', f'_diag_{tag}.pt')
    torch.save({'model': m}, tmp)
    return RTDETR(tmp)


def predict_frames(m, frames, images_dir):
    import cv2
    preds = []
    with torch.no_grad():
        for f in frames:
            im = cv2.imread(os.path.join(images_dir, f[3]))
            r = m.predict(im, imgsz=640, conf=0.001, max_det=300, verbose=False)[0]
            b = r.boxes
            if b is None or len(b) == 0:
                preds.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                              np.zeros(0, np.int64)))
            else:
                preds.append((b.xyxy.cpu().numpy().astype(np.float32),
                              b.conf.cpu().numpy().astype(np.float32),
                              b.cls.cpu().numpy().astype(np.int64)))
    return preds


def ap_of(preds, gt, frames):
    """与 eval_sap verify 同口径: 同帧对同帧, 逐类积分再平均"""
    all_d = []
    for i, (bx, sc, cl) in enumerate(preds):
        for j in range(len(sc)):
            all_d.append((sc[j], cl[j], i, bx[j]))
    all_d.sort(key=lambda x: -x[0])
    aps = []
    for thr in IOU_THRS:
        per_cls = []
        for c in range(NC):
            n_gt = sum(int((gt[frames[i][3]][1] == c).sum()) for i in range(len(frames)))
            if n_gt == 0:
                continue
            dets = [d for d in all_d if d[1] == c]
            tp = np.zeros(len(dets), bool)
            by_frame = {}
            for k, d in enumerate(dets):
                by_frame.setdefault(d[2], []).append(k)
            for fi, ks in by_frame.items():
                gb, gc = gt[frames[fi][3]]
                m = gc == c
                if not m.any():
                    continue
                sub = np.array(ks)
                iou = iou_matrix(np.stack([dets[k][3] for k in sub]), gb[m])
                tp[sub] = greedy_tp(iou, thr)
            csum = np.cumsum(tp)
            rec = csum / n_gt
            prec = csum / np.arange(1, len(dets) + 1)
            per_cls.append(voc_ap(rec, prec))
        aps.append(float(np.mean(per_cls)) if per_cls else 0.0)
    return aps[0], float(np.mean(aps))


def main():
    frames, _, _ = build_timeline(os.path.join(DATA, 'stream_index.json'))
    frames = frames[:N_FRAMES]
    gt = load_gt(os.path.join(DATA, 'annotations_val.json'))
    images_dir = os.path.join(DATA, 'images')

    for tag, ckpt, struct in [
        ('A 初始 3L(未训练)', CKPT_INIT, {'ndl': 3, 'nq': 300, 'nd': 0}),
        ('B b1a0 训后', CKPT_B1A0, {'ndl': 3, 'nq': 300, 'nd': 0}),
    ]:
        m = as_api(build_and_load(ckpt, struct), tag.split()[0])
        preds = predict_frames(m, frames, images_dir)
        ap50, ap = ap_of(preds, gt, frames)
        print(f'{tag}: AP50={ap50:.4f} AP={ap:.4f}  ({N_FRAMES} 帧)', flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
