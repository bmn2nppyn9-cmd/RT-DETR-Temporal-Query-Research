# -*- coding: utf-8 -*-
"""
verify_query_timing.py —— 查询数的延迟复核(交错测量, 排除热漂移)
=================================================================
为什么重测:
    eval_query_sweep.py 内顺序测各档延迟, 结果 150 查询(20.58ms) 反而比
    300 查询(18.67ms) 更慢 —— 与物理矛盾。
    根因同 depth sweep: 顺序测量下 GPU 逐渐升温, 后测的档位吃亏。
    本机功耗锁 140W, 对热极敏感(实测预热 60C 时 SM 2707MHz, 25 轮后 77C 降到 2685MHz)。

做法: 与 verify_depth_timing.py 一致的 round-robin 交错采样 ——
      每轮把所有查询档位各测一遍, 重复 N 轮, 热漂移被均摊到所有档位。

用法:
    python verify_query_timing.py --queries 300 250 200 150 100 75 50 --rounds 25
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np, torch

sys.path.insert(0, '.')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def log(m): print(m, flush=True)


def find_decoder(model):
    for mod in model.modules():
        if type(mod).__name__ == 'RTDETRDecoder' and hasattr(mod, 'num_queries'):
            return mod
    return None


def gpu_stat():
    try:
        import subprocess
        o = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,clocks.current.sm',
                            '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
        return o.stdout.strip()
    except Exception:
        return 'n/a'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=r'runs\detect\runs_rtdetr\r18_640_sub20_b4\weights\best.pt')
    ap.add_argument('--queries', nargs='+', type=int, default=[300, 250, 200, 150, 100, 75, 50])
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--rounds', type=int, default=25)
    ap.add_argument('--runs', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=60)
    ap.add_argument('--out', default='out_query_timing.json')
    args = ap.parse_args()

    from ultralytics import RTDETR
    m = RTDETR(args.model)
    dec = find_decoder(m.model)
    inner = dec.decoder
    nq_orig = dec.num_queries
    inner.eval_idx = args.depth - 1
    log(f'num_layers={dec.num_decoder_layers}  固定深度={args.depth}  '
        f'原 num_queries={nq_orig}')

    net = m.model.cuda().eval()
    x = torch.rand(1, 3, args.imgsz, args.imgsz, device='cuda')

    log(f'预热 {args.warmup} 次...  GPU: {gpu_stat()}')
    with torch.no_grad():
        for _ in range(args.warmup):
            net(x)
    torch.cuda.synchronize()
    log(f'预热完成 GPU: {gpu_stat()}')

    def meas(q, n):
        dec.num_queries = q
        with torch.no_grad():
            for _ in range(3):
                net(x)
        torch.cuda.synchronize()
        ts = []
        with torch.no_grad():
            for _ in range(n):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                net(x)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
        return ts

    samples = {q: [] for q in args.queries}
    log('')
    log(f'交错测量: {args.rounds} 轮 × {len(args.queries)} 档 × {args.runs} 次'
        f' = 每档 {args.rounds*args.runs} 样本')
    t0 = time.time()
    for r in range(args.rounds):
        for q in args.queries:
            samples[q].extend(meas(q, args.runs))
        if (r + 1) % 5 == 0:
            log(f'  第 {r+1}/{args.rounds} 轮  ({time.time()-t0:.0f}s)  GPU: {gpu_stat()}')

    dec.num_queries = nq_orig

    stats = {}
    for q in args.queries:
        v = np.array(samples[q])
        stats[q] = dict(n=len(v), median=float(np.median(v)), mean=float(v.mean()),
                        std=float(v.std()), p95=float(np.percentile(v, 95)),
                        p99=float(np.percentile(v, 99)))

    log('')
    log(f"{'查询数':>7}{'样本':>7}{'中位(ms)':>11}{'标准差':>9}{'P95':>9}{'P99':>9}")
    for q in args.queries:
        s = stats[q]
        log(f"{q:>7}{s['n']:>7}{s['median']:>11.2f}{s['std']:>9.2f}"
            f"{s['p95']:>9.2f}{s['p99']:>9.2f}")

    base = stats[args.queries[0]]['median']
    log('')
    log(f'=== 相对 {args.queries[0]} 查询 ===')
    inv = []
    prev = None
    for q in args.queries:
        sav = (1 - stats[q]['median'] / base) * 100
        log(f'  {q:>4} 查询: {stats[q]["median"]:6.2f} ms   省时 {sav:+6.1f}%')
        if prev and stats[q]['median'] < prev[1]:
            inv.append((prev[0], q))
        prev = (q, stats[q]['median'])
    log('')
    if not inv:
        log('结论: 延迟随查询数单调递减, 符合预期。')
    else:
        log(f'警告: 仍有逆序 {inv}, 需排查')

    json.dump(dict(depth=args.depth, interleaved=True, rounds=args.rounds,
                   stats={str(k): v for k, v in stats.items()}),
              open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    log(f'已写入 {args.out}')


if __name__ == '__main__':
    main()
