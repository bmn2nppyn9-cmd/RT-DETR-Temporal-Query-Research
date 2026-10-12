# -*- coding: utf-8 -*-
"""
eval_query_sweep.py —— decoder 查询数量的精度-时延扫描
=====================================================
背景:
    AQS-IDETR (Image and Vision Computing 2026, DOI 10.1016/j.imavis.2026.106097)
    报告: COCO 上把 decoder 查询剪掉 50%, AP 只掉 ~0.1%, 评估时间降 30-35%。
    该文是训练无关的推理期方法, 在 RT-DETR/v2/v3/DINO-DETR 上验证。

    本项目要回答的是: 在 Argoverse-HD (车载视频, 每帧中位 15 个目标,
    比 COCO 的 ~7 个更密集) 上, 砍查询的代价是多少?

本脚本测什么:
    RT-DETR 的 decoder 查询数由 RTDETRDecoder.num_queries 决定
    (_get_decoder_input 里对 enc_score 做 topk(k=num_queries),
     dec_score_head/dec_bbox_head 是逐 query 的 MLP, 与 query 数无关)。
    因此【改 num_queries 即可在推理期改变查询数, 无需重训】——
    这正好复现了 AQS-IDETR "training-free" 的设定。

与 AQS-IDETR 的差异(报告结论时必须说明):
    本脚本是【固定 N】: 一开始就只用 N 个查询, 跑满全部层。
    AQS 是【逐层渐进剪枝】: 前几层仍保留较多查询, 每层之后淘汰低分者。
    固定 N 相当于 AQS 的"更激进/更差"版本(起点就更少),
    因为 AQS 至少在浅层保留了完整查询来精修。
    所以: 若固定 N 的精度损失已经很小, 则渐进剪枝必然更小;
          若固定 N 损失较大, 渐进剪枝可能显著更好 —— 那时需要单独实现验证。

用法:
    python eval_query_sweep.py --queries 300 150 --fraction 0.2    # 快速自检
    python eval_query_sweep.py --queries 300 200 150 100           # 全量 val
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def log(m):
    print(m, flush=True)


def find_decoder(model):
    for mod in model.modules():
        if type(mod).__name__ == 'RTDETRDecoder' and hasattr(mod, 'num_queries'):
            return mod
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',
                    default=r'runs\detect\runs_rtdetr\r18_640_sub20_b4\weights\best.pt')
    ap.add_argument('--data', default=r'cfg/argoverse-hd-local.yaml')
    ap.add_argument('--queries', nargs='+', type=int,
                    default=[300, 200, 150, 100])
    ap.add_argument('--depth', type=int, default=6,
                    help='固定解码深度, 只扫查询数')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--fraction', type=float, default=1.0,
                    help='val 使用比例(调试用 0.2)')
    ap.add_argument('--timing-runs', type=int, default=30,
                    help='每档测延迟的前向次数(合成输入)')
    ap.add_argument('--out', default='out_query_sweep.json')
    args = ap.parse_args()

    from ultralytics import RTDETR
    m = RTDETR(args.model)
    dec = find_decoder(m.model)
    if dec is None:
        log('[错误] 找不到 RTDETRDecoder')
        sys.exit(1)
    nq_orig = dec.num_queries
    log(f'RTDETRDecoder: num_queries={nq_orig}  num_layers={dec.num_decoder_layers}')

    # 固定解码深度
    inner = dec.decoder
    log(f'固定解码深度 = {args.depth} 层 (eval_idx={args.depth-1})')
    inner.eval_idx = args.depth - 1

    net = m.model.cuda().eval()
    x = torch.rand(1, 3, args.imgsz, args.imgsz, device='cuda')

    results = []
    log('')
    log(f"{'查询数':>7}{'mAP50':>10}{'mAP50-95':>11}{'Precision':>11}"
        f"{'Recall':>9}{'延迟(ms)':>11}{'省时':>9}")
    base_lat = None
    for q in args.queries:
        dec.num_queries = q
        r = m.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=0,
                  workers=3, split='val', verbose=False, plots=False,
                  fraction=args.fraction)
        b = r.box

        # 测纯前向延迟(合成输入, 中位)
        with torch.no_grad():
            for _ in range(5):
                net(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.timing_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                net(x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        lat = float(np.median(ts))
        if base_lat is None:
            base_lat = lat
        # val 会让模型切到 eval 并可能改变 device 状态, 重新取回
        net = m.model.cuda().eval()

        results.append(dict(queries=q, map50=float(b.map50), map=float(b.map),
                            precision=float(b.mp), recall=float(b.mr),
                            latency_ms=lat))
        log(f"{q:>7}{b.map50:>10.4f}{b.map:>11.4f}{b.mp:>11.4f}{b.mr:>9.4f}"
            f"{lat:>11.2f}{(1-lat/base_lat)*100:>8.1f}%")

    dec.num_queries = nq_orig

    # 汇总
    ref = results[0]
    log('')
    log(f'=== 相对 {ref["queries"]} 查询的差距 ===')
    log(f"{'查询数':>7}{'ΔmAP50(pp)':>13}{'ΔmAP50-95(pp)':>15}"
        f"{'省时(%)':>10}{'保留率':>9}")
    for r in results:
        log(f"{r['queries']:>7}{(r['map50']-ref['map50'])*100:>+13.2f}"
            f"{(r['map']-ref['map'])*100:>+15.2f}"
            f"{(1-r['latency_ms']/ref['latency_ms'])*100:>10.1f}"
            f"{100*r['queries']/ref['queries']:>8.0f}%")

    # 与 AQS-IDETR 的报告值对照
    half = next((r for r in results if r['queries'] == ref['queries'] // 2), None)
    log('')
    log('=' * 68)
    log('对照 AQS-IDETR 报告值 (COCO, 剪枝 50%): AP -0.1%, 评估时间 -30~35%')
    if half:
        log(f'本项目 Argoverse-HD, 查询 300 -> {half["queries"]} (50%):')
        log(f'  mAP50 {ref["map50"]:.4f} -> {half["map50"]:.4f} '
            f'({(half["map50"]-ref["map50"])*100:+.2f} pp, '
            f'相对 {(1-half["map50"]/ref["map50"])*100:+.1f}%)')
        log(f'  延迟 {ref["latency_ms"]:.2f} -> {half["latency_ms"]:.2f} ms '
            f'({(1-half["latency_ms"]/ref["latency_ms"])*100:+.1f}%)')
        log(f'  注: 本测试为固定 N 剪枝(起始即少), AQS 为逐层渐进剪枝,')
        log(f'      渐进剪枝的精度损失应更小。')
    log('=' * 68)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(dict(depth=args.depth, fraction=args.fraction, results=results),
                  f, ensure_ascii=False, indent=2)
    log(f'已写入 {args.out}')


if __name__ == '__main__':
    main()
