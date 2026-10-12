# -*- coding: utf-8 -*-
"""
record_b1a.py —— B1a 训练后流式录制(21 场景全量 val)
======================================================
与 record_qd_t.py 同协议(batch=1 逐帧 predict, 场景边界 reset, 输出 npz
供 eval_full_metrics.py / eval_qd_t 链消费)。差异:
  - 每个配置加载自己的 B1a 训练 checkpoint;
  - temporal 配置从 ckpt 的 b1a_temporal_spec 读取(source_depth/injection_layer/M),
    source adapter 由 eval 注入路径自动应用(与训练同口径);
  - 对照组(b1a0/b1ac25)temporal 关闭; b1ac25 的输出 cap 由 ckpt 内
    b1a_eval_topk=300 自动生效。

用法:
  python record_b1a.py --configs b1a0 b1at33 --run-dir runs --out records_e1
  python record_b1a.py --configs b1at33 --limit-scenes 2        # smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqinfra.runtime import activate_backend, DATA_ROOT, PAIR_DIR, TRAIN_DIR, RECORD_DIR, EVAL_DIR, INITIAL_WEIGHTS, MODEL_YAML, ROOT
activate_backend()

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = str(DATA_ROOT / 'argoverse-hd')

CONFIGS = {
    'b1a0': {'temporal': False},
    'b1ac25': {'temporal': False},
    'b1at31': {'temporal': True},
    'b1at23': {'temporal': True},
    'b1at33': {'temporal': True},
}


def log(m):
    print(m, flush=True)


def load_bgr(p):
    import cv2
    im = cv2.imread(p)
    if im is None:
        raise FileNotFoundError(p)
    return im


def find_head(model):
    from ultralytics.nn.modules.head import RTDETRDecoder
    for mod in model.modules():
        if isinstance(mod, RTDETRDecoder):
            return mod
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', choices=list(CONFIGS), default=list(CONFIGS))
    ap.add_argument('--run-dir', default=str(TRAIN_DIR))
    ap.add_argument('--ckpt-name', default=os.path.join('weights', 'best.pt'))
    ap.add_argument('--stream-index', default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--images', default=os.path.join(DATA, 'images'))
    ap.add_argument('--out', default=str(RECORD_DIR))
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0')
    ap.add_argument('--limit-frames', type=int, default=0)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--max-det', type=int, default=300)
    ap.add_argument('--limit-scenes', type=int, default=0)
    ap.add_argument('--prefetch', type=int, default=8)
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    idx = json.load(open(args.stream_index, encoding='utf-8'))
    scenes = idx['scenes'][:args.limit_scenes] if args.limit_scenes else idx['scenes']
    n_frames = sum(s['n_frames'] for s in scenes)
    if not scenes or not scenes[0]['frames']:
        raise ValueError('Empty stream index')
    log(f'流式索引: {len(scenes)} 场景 / {n_frames} 帧')

    from ultralytics import RTDETR
    for name in args.configs:
        ckpt = os.path.join(args.run_dir, name, args.ckpt_name)
        out_path = os.path.join(args.out, f'preds_{name}.npz')
        if args.skip_existing and os.path.exists(out_path):
            log(f'[跳过] {out_path} 已存在')
            continue
        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)

        m = RTDETR(ckpt)
        head = find_head(m.model)
        assert head is not None, '找不到 RTDETRDecoder'
        assert head.decoder.eval_idx == 2, f'eval_idx={head.decoder.eval_idx}, 应为 2'
        head.set_temporal_enabled(False)
        head.time_decoder_cuda = False
        head.set_query_overrides(encoder_queries=300, temporal_queries=0,
                                 output_topk=300, motion_alpha=0.0)

        temporal = CONFIGS[name]['temporal']
        spec = getattr(head, 'b1a_temporal_spec', None)
        if temporal:
            assert spec is not None, f'{name} ckpt 无 b1a_temporal_spec'
            head.set_temporal_config(source_depth=spec['source_depth'],
                                     injection_layer=spec['injection_layer'],
                                     fuse_after=False)
            head.set_query_overrides(temporal_queries=spec['M'])
        log('=' * 60)
        log(f'[{name}] ckpt={os.path.basename(os.path.dirname(os.path.dirname(ckpt)))} '
            f'temporal={temporal} spec={spec}')
        # warmup (temporal off)
        warm = load_bgr(os.path.join(args.images, scenes[0]['frames'][0]['file']))
        for _ in range(args.warmup):
            m.predict(warm, imgsz=args.imgsz, conf=args.conf, max_det=args.max_det,
                      device=args.device, verbose=False)
        if temporal:
            head.set_temporal_enabled(True)
            head.time_decoder_cuda = False

        scene_arr, fidx_arr, t_arr, lat_arr, dec_arr, nq_arr = [], [], [], [], [], []
        boxes, scores, clss, qidxs, srcs, sdeps, injs, ptr = [], [], [], [], [], [], [], [0]
        t_start = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=args.prefetch) as pool:
            for s in scenes:
                head.reset_temporal_state()
                files = [os.path.join(args.images, fr['file']) for fr in s['frames']]
                ts = [fr['t_ns'] for fr in s['frames']]
                if args.limit_frames:
                    files = files[:args.limit_frames]
                    ts = ts[:args.limit_frames]
                for base in range(0, len(files), 32):
                    part = files[base:base + 32]
                    imgs = list(pool.map(load_bgr, part))
                    for k, im in enumerate(imgs):
                        t0 = time.perf_counter()
                        r = m.predict(im, imgsz=args.imgsz, conf=args.conf,
                                      max_det=args.max_det, device=args.device, verbose=False)[0]
                        lat = (time.perf_counter() - t0) * 1000.0
                        b = r.boxes
                        n = 0 if b is None else len(b)
                        meta = head._last_topk_meta
                        keep = (meta['scores'][0] > args.conf).nonzero(as_tuple=True)[0]
                        keep = keep[:args.max_det]
                        if n:
                            boxes.append(b.xyxy.cpu().numpy().astype(np.float32))
                            scores.append(b.conf.cpu().numpy().astype(np.float32))
                            clss.append(b.cls.cpu().numpy().astype(np.uint8))
                            qidxs.append(meta['query_idx'][0][keep].cpu().numpy().astype(np.int16))
                            srcs.append(meta['source'][0][keep].cpu().numpy().astype(np.int8))
                            sdeps.append(meta['source_depth'][0][keep].cpu().numpy().astype(np.int8))
                            injs.append(meta['injection_layer'][0][keep].cpu().numpy().astype(np.int8))
                        ptr.append(ptr[-1] + n)
                        scene_arr.append(s['scene_id'])
                        fidx_arr.append(base + k)
                        t_arr.append(ts[base + k])
                        lat_arr.append(lat)
                        dec_arr.append(head._last_decoder_ms)
                        nq_arr.append(head._last_decoder_query_count)
                        done += 1
                    if done % 1024 == 0 or done == n_frames:
                        el = time.time() - t_start
                        log(f'  {done}/{n_frames}  {el:.0f}s  '
                            f'{done / max(el, 1e-9):.1f} 帧/秒')

        np.savez_compressed(
            out_path,
            scene=np.array(scene_arr, np.int16),
            frame_idx=np.array(fidx_arr, np.int16),
            t_ns=np.array(t_arr, np.int64),
            compute_ms=np.array(lat_arr, np.float32),
            dec_ms=np.array(dec_arr, np.float32),
            box=(np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32)),
            score=(np.concatenate(scores) if scores else np.zeros(0, np.float32)),
            cls=(np.concatenate(clss) if clss else np.zeros(0, np.uint8)),
            qidx=(np.concatenate(qidxs) if qidxs else np.zeros(0, np.int16)),
            src=(np.concatenate(srcs) if srcs else np.zeros(0, np.int8)),
            src_depth=(np.concatenate(sdeps) if sdeps else np.zeros(0, np.int8)),
            inj_layer=(np.concatenate(injs) if injs else np.zeros(0, np.int8)),
            ptr=np.array(ptr, np.int32),
            nq_dec=np.array(nq_arr, np.int16),
            config=np.str_(name),
        )
        med = float(np.median(lat_arr))
        log(f'  完成 {name}: {done} 帧, 推理中位 {med:.2f}ms -> {out_path}')
        head.set_temporal_enabled(False)
        head.time_decoder_cuda = False
        del m
        import torch
        torch.cuda.empty_cache()

    log('全部录制完成。')


if __name__ == '__main__':
    main()
