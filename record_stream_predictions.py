# -*- coding: utf-8 -*-
"""
record_stream_predictions.py —— 录制流式推理的逐帧预测与真实耗时
==================================================================
为什么先录再算:
    sAP 的延迟惩罚依赖"每个预测的产出时刻"。若每换一个延迟档位就重跑一遍
    推理(3 深度 × N 档 × 15062 帧), 时间上不可接受。因此:
        先按真实时序跑一遍推理, 把逐帧预测 + 实测单帧耗时落盘;
        之后所有延迟档位都在离线调度器里重算(秒级)。

关键设计:
  1. 按 stream_index 的场景顺序逐帧推进 —— 时序不能乱, 否则"上一帧"无意义
  2. 只对 predict() 计时而把图片解码排除在外 —— 流式系统里帧来自相机,
     解码与推理可以流水线重叠, 不应算进模型的计算延迟
  3. 低置信阈值(0.001) + max_det=300 —— PR 曲线需要完整的低分预测
  4. 逐深度存盘, 中途可断点续跑

输出(每个深度一个 npz):
    scene      (Nf,)   int16   场景号
    frame_idx  (Nf,)   int16   场景内帧序号
    t_ns       (Nf,)   int64   帧捕获时刻(纳秒, 来自文件名)
    compute_ms (Nf,)   float32 该帧实测推理耗时
    box        (Nd,4)  float32 xyxy 像素坐标
    score      (Nd,)   float32
    cls        (Nd,)   uint8
    ptr        (Nf+1,) int32   第 i 帧的预测在 box/score/cls 中的切片 [ptr[i]:ptr[i+1]]

用法:
    python record_stream_predictions.py --depths 1 3 6
    python record_stream_predictions.py --depths 3 --limit-scenes 2   # 试跑
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'


def log(m):
    print(m, flush=True)


def load_bgr(path):
    """读图为 BGR ndarray —— ultralytics 的 numpy 输入约定为 BGR"""
    with Image.open(path) as im:
        return np.array(im.convert('RGB'))[:, :, ::-1].copy()


def find_decoder(model):
    """定位 DeformableTransformerDecoder(RTDETRDecoder.decoder)"""
    for mod in model.modules():
        if type(mod).__name__ == 'RTDETRDecoder' and hasattr(mod, 'decoder'):
            return mod.decoder
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',
                    default=r'runs\detect\runs_rtdetr\r18_640_sub20_b4\weights\best.pt')
    ap.add_argument('--stream-index', default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--images', default=os.path.join(DATA, 'images'))
    ap.add_argument('--depths', nargs='+', type=int, default=[1, 3, 6])
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.001, help='低阈值以保留完整 PR 曲线')
    ap.add_argument('--max-det', type=int, default=300)
    ap.add_argument('--out', default='stream_preds')
    ap.add_argument('--limit-scenes', type=int, default=0, help='只跑前 N 个场景(试跑)')
    ap.add_argument('--prefetch', type=int, default=8, help='图片解码预取线程数')
    ap.add_argument('--skip-existing', action='store_true', help='已存在的深度直接跳过')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.stream_index, encoding='utf-8') as f:
        idx = json.load(f)
    scenes = idx['scenes']
    if args.limit_scenes:
        scenes = scenes[:args.limit_scenes]
    n_frames = sum(s['n_frames'] for s in scenes)
    log(f'流式索引: {len(scenes)} 场景 / {n_frames} 帧')

    from ultralytics import RTDETR
    log(f'加载模型 {args.model}')
    m = RTDETR(args.model)
    decoder = find_decoder(m.model)
    if decoder is None:
        log('[错误] 找不到 DeformableTransformerDecoder')
        sys.exit(1)
    log(f'decoder: num_layers={decoder.num_layers}  原 eval_idx={decoder.eval_idx}')
    saved_idx = decoder.eval_idx

    for depth in args.depths:
        out_path = os.path.join(args.out, f'preds_d{depth}.npz')
        if args.skip_existing and os.path.exists(out_path):
            log(f'\n[跳过] {out_path} 已存在')
            continue

        decoder.eval_idx = depth - 1
        log('')
        log('=' * 60)
        log(f'录制 depth={depth} 层  -> {out_path}')
        log('=' * 60)

        scene_arr, fidx_arr, t_arr, lat_arr = [], [], [], []
        boxes, scores, clss, ptr = [], [], [], [0]
        t_start = time.time()
        done = 0

        with ThreadPoolExecutor(max_workers=args.prefetch) as pool:
            for s in scenes:
                files = [os.path.join(args.images, fr['file']) for fr in s['frames']]
                ts = [fr['t_ns'] for fr in s['frames']]
                # 分块预取: 解码在后台线程进行, 推理在主线程按序推进
                chunk = 32
                for base in range(0, len(files), chunk):
                    part = files[base:base + chunk]
                    imgs = list(pool.map(load_bgr, part))
                    for k, im in enumerate(imgs):
                        # 只计时 predict 本身, 不含图片解码
                        t0 = time.perf_counter()
                        r = m.predict(im, imgsz=args.imgsz, conf=args.conf,
                                      max_det=args.max_det, device=0, verbose=False)[0]
                        lat = (time.perf_counter() - t0) * 1000.0

                        b = r.boxes
                        n = 0 if b is None else len(b)
                        if n:
                            boxes.append(b.xyxy.cpu().numpy().astype(np.float32))
                            scores.append(b.conf.cpu().numpy().astype(np.float32))
                            clss.append(b.cls.cpu().numpy().astype(np.uint8))
                        ptr.append(ptr[-1] + n)

                        scene_arr.append(s['scene_id'])
                        fidx_arr.append(base + k)
                        t_arr.append(ts[base + k])
                        lat_arr.append(lat)
                        done += 1
                    el = time.time() - t_start
                    if done % 512 == 0 or done == n_frames:
                        log(f'  {done}/{n_frames}  {el:.0f}s  '
                            f'{done/max(el,1e-9):.1f} 帧/秒  '
                            f'近帧耗时 {lat:.1f}ms')

        box_all = (np.concatenate(boxes) if boxes
                   else np.zeros((0, 4), np.float32))
        np.savez_compressed(
            out_path,
            scene=np.array(scene_arr, np.int16),
            frame_idx=np.array(fidx_arr, np.int16),
            t_ns=np.array(t_arr, np.int64),
            compute_ms=np.array(lat_arr, np.float32),
            box=box_all,
            score=(np.concatenate(scores) if scores else np.zeros(0, np.float32)),
            cls=(np.concatenate(clss) if clss else np.zeros(0, np.uint8)),
            ptr=np.array(ptr, np.int32),
            depth=np.int32(depth),
        )
        med = float(np.median(lat_arr))
        p95 = float(np.percentile(lat_arr, 95))
        log(f'  完成: {done} 帧, {len(box_all)} 个预测, '
            f'推理耗时 中位 {med:.2f}ms / P95 {p95:.2f}ms')
        log(f'  已写入 {out_path}')

    decoder.eval_idx = saved_idx
    log('')
    log('全部深度录制完成。下一步: python eval_sap.py')


if __name__ == '__main__':
    main()
