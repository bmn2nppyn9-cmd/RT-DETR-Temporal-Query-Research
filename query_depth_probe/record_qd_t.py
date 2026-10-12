# -*- coding: utf-8 -*-
"""record_qd_t.py —— QD-T 阶段: Temporal Query 注入深度扫描录制
==============================================================
矩阵 (A0 无 temporal; T 命名 = 前帧 source depth + 注入层):

    A0  无 temporal, 3 层 300 current                       (基线)
    T31 S3 -> L1 前, 运行 3 层 (现有 B0 机制, 最强深度错位)
    T01 S0 -> L1 前, 运行 3 层 (同阶段对齐: encoder 级进入 L1)
    T12 S1 -> L2 前, 运行 2 层 (同阶段对齐: L1 输出进入 L2)
    T23 S2 -> L3 前, 运行 1 层 (同阶段对齐: L2 输出进入 L3)
    T33 S3 -> L3 前, 运行 1 层 (成熟 Query 只做一次当前帧 refine)
    T3O S3 -> Decoder 后融合, 运行 0 层 (旧结果直接融合下限)

每档 M ∈ {25, 75}。选择统一用前帧 L3 分数 (score3) top-M, 各深度 S0..S3 按同一
query index 汇集, 不同 source depth 不选不同的 Query 集合。

每帧记录: 输出 box/score/cls + 每输出条 qidx/src/src_depth/inj_layer
+ decoder 输入宽度 nq_dec + decoder-only CUDA Event 计时 dec_ms + predict wall。

协议: predictor warmup 在 temporal 关闭状态完成, 之后按配置 reset 再录制。
首帧无 temporal state (此时 T 组 ≡ A0)。

用法:
    python record_qd_t.py --limit-scenes 2                     # smoke 全矩阵
    python record_qd_t.py --configs A0 T31 T23 T33 --m 75     # 全量 (确认后)
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

sys.path.insert(0, r'C:\Users\TONY\ultralytics')
import ultralytics  # noqa: E402
assert ultralytics.__file__.lower().startswith(r'c:\users\tony\ultralytics\ultralytics'), \
    f'导入路径错误: {ultralytics.__file__}'

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
CKPT = r'D:\lunwen\自研\runs\detect\runs_rtdetr\r18_640_sub20_b4\weights\best.pt'
OUT = r'D:\lunwen\自研\query_depth_probe\stream_preds'

T_MATRIX = {
    'A0': dict(temporal=False, src=3, inj=0, fuse=False),
    'T31': dict(temporal=True, src=3, inj=0, fuse=False),
    'T01': dict(temporal=True, src=0, inj=0, fuse=False),
    'T12': dict(temporal=True, src=1, inj=1, fuse=False),
    'T23': dict(temporal=True, src=2, inj=2, fuse=False),
    'T33': dict(temporal=True, src=3, inj=2, fuse=False),
    'T3O': dict(temporal=True, src=3, inj=2, fuse=True),
}


def log(m):
    print(m, flush=True)


def load_bgr(path):
    with Image.open(path) as im:
        return np.array(im.convert('RGB'))[:, :, ::-1].copy()


def find_head(model):
    for mod in model.modules():
        if type(mod).__name__ == 'RTDETRDecoder' and hasattr(mod, 'decoder'):
            return mod
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stream-index', default=os.path.join(DATA, 'stream_index.json'))
    ap.add_argument('--images', default=os.path.join(DATA, 'images'))
    ap.add_argument('--configs', nargs='+', default=list(T_MATRIX))
    ap.add_argument('--m', nargs='+', type=int, default=[25, 75])
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--max-det', type=int, default=300)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--limit-scenes', type=int, default=0)
    ap.add_argument('--prefetch', type=int, default=8)
    ap.add_argument('--skip-existing', action='store_true')
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
    m = RTDETR(CKPT)
    head = find_head(m.model)
    if head is None:
        log('[错误] 找不到 RTDETRDecoder')
        sys.exit(1)
    head.decoder.eval_idx = 2
    head.set_temporal_enabled(False)
    head.set_query_overrides(encoder_queries=300, temporal_queries=0, output_topk=300, motion_alpha=0.0)
    head.time_decoder_cuda = False

    # 协议: predictor warmup 在 temporal 关闭状态完成
    warm_img = load_bgr(os.path.join(args.images, scenes[0]['frames'][0]['file']))
    _ = m.predict(warm_img, imgsz=args.imgsz, conf=args.conf, max_det=args.max_det,
                  device=0, verbose=False)[0]
    log('warmup 完成 (temporal off)')

    tags = []
    for cfg_name in args.configs:
        cfg = T_MATRIX[cfg_name]
        if cfg_name == 'A0':
            tags.append(('A0', cfg, 0))
        else:
            for mm in args.m:
                tags.append((cfg_name, cfg, mm))

    for cfg_name, cfg, mm in tags:
        tag = f'{cfg_name.lower()}m{mm}' if cfg_name != 'A0' else 'a0'
        out_path = os.path.join(args.out, f'preds_{tag}.npz')
        if args.skip_existing and os.path.exists(out_path):
            log(f'\n[跳过] {out_path} 已存在')
            continue

        # 配置应用: 先彻底关闭并清空, 再按组设置
        head.set_temporal_enabled(False)
        head.time_decoder_cuda = False
        head.set_query_overrides(encoder_queries=300, temporal_queries=mm, output_topk=300, motion_alpha=0.0)
        head.set_temporal_config(source_depth=cfg['src'], injection_layer=cfg['inj'], fuse_after=cfg['fuse'])
        if cfg['temporal']:
            head.set_temporal_enabled(True)
        head.time_decoder_cuda = cfg['temporal']  # 仅 temporal 组测 decoder-only 时间
        log('')
        log('=' * 60)
        log(f'录制 {tag} (src={cfg["src"]} inj={cfg["inj"]} fuse={cfg["fuse"]} M={mm}) -> {out_path}')
        log(f'  {head.temporal_status()}')
        log('=' * 60)

        scene_arr, fidx_arr, t_arr, lat_arr, dec_arr, nq_arr = [], [], [], [], [], []
        boxes, scores, clss, qidxs, srcs, sdeps, injs, ptr = [], [], [], [], [], [], [], [0]
        t_start = time.time()
        done = 0
        src_mismatch = 0

        with ThreadPoolExecutor(max_workers=args.prefetch) as pool:
            for s in scenes:
                # 场景边界: 任何历史状态不得跨场景
                head.reset_temporal_state()
                files = [os.path.join(args.images, fr['file']) for fr in s['frames']]
                ts = [fr['t_ns'] for fr in s['frames']]
                chunk = 32
                for base in range(0, len(files), chunk):
                    part = files[base:base + chunk]
                    imgs = list(pool.map(load_bgr, part))
                    for k, im in enumerate(imgs):
                        t0 = time.perf_counter()
                        r = m.predict(im, imgsz=args.imgsz, conf=args.conf,
                                      max_det=args.max_det, device=0, verbose=False)[0]
                        lat = (time.perf_counter() - t0) * 1000.0

                        b = r.boxes
                        n = 0 if b is None else len(b)
                        meta = head._last_topk_meta
                        # query 级标注: 复现 predictor 的 conf>thr + max_det 过滤, 与输出逐一对齐
                        keep = (meta['scores'][0] > args.conf).nonzero(as_tuple=True)[0]
                        keep = keep[:args.max_det]
                        src = meta['source'][0][keep].cpu().numpy().astype(np.int8)
                        if len(src) != n:
                            src_mismatch += 1
                            src = np.full(n, -1, np.int8)
                        qidx = meta['query_idx'][0][keep].cpu().numpy().astype(np.int16)
                        sdep = meta['source_depth'][0][keep].cpu().numpy().astype(np.int8)
                        inj = meta['injection_layer'][0][keep].cpu().numpy().astype(np.int8)
                        if n:
                            boxes.append(b.xyxy.cpu().numpy().astype(np.float32))
                            scores.append(b.conf.cpu().numpy().astype(np.float32))
                            clss.append(b.cls.cpu().numpy().astype(np.uint8))
                            qidxs.append(qidx)
                            srcs.append(src)
                            sdeps.append(sdep)
                            injs.append(inj)
                        ptr.append(ptr[-1] + n)

                        scene_arr.append(s['scene_id'])
                        fidx_arr.append(base + k)
                        t_arr.append(ts[base + k])
                        lat_arr.append(lat)
                        dec_arr.append(head._last_decoder_ms)
                        nq_arr.append(head._last_decoder_query_count)
                        done += 1
                    el = time.time() - t_start
                    if done % 512 == 0 or done == n_frames:
                        log(f'  {done}/{n_frames}  {el:.0f}s  '
                            f'{done / max(el, 1e-9):.1f} 帧/秒  近帧耗时 {lat:.1f}ms')

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
            config=np.str_(tag),
        )
        med = float(np.median(lat_arr))
        p95 = float(np.percentile(lat_arr, 95))
        dmed = float(np.median(dec_arr)) if cfg['temporal'] else 0.0
        log(f'  完成: {done} 帧, {len(boxes) and sum(len(x) for x in boxes)} 条预测, '
            f'推理 中位 {med:.2f}ms / P95 {p95:.2f}ms, decoder-only 中位 {dmed:.3f}ms, '
            f'src 对齐失败 {src_mismatch} 帧')
        log(f'  已写入 {out_path}')

    # 收尾: 恢复默认, 不留状态
    head.set_temporal_enabled(False)
    head.time_decoder_cuda = False
    head.set_query_overrides(encoder_queries=300, temporal_queries=0, output_topk=300)
    log('')
    log('全部配置录制完成。')


if __name__ == '__main__':
    main()
