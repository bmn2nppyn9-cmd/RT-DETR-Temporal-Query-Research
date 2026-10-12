# -*- coding: utf-8 -*-
"""
verify_depth_timing.py —— 复核 decoder 层数与延迟的关系(排除热漂移)
====================================================================
为什么要重测:
    eval_depth_sweep.py 里测出 3 层(25.28ms) 比 6 层(22.51ms) 还慢 12.3%,
    与"层数越少越快"矛盾。可疑根因是该脚本在每个档位都先跑一次全量 val
    (15062 张)、再测延迟, 于是:
      1) GPU 处于热态, 且各档位热积累程度不同
      2) 三个档位顺序执行, 热漂移会系统性偏袒先跑的档位
    本机 5070 Ti Laptop 功耗锁 140W、SM 仅 907/3090MHz, 对热/功耗极敏感,
    这种顺序测量不可靠。

已知的两种错误测法(不要重复):
    A. monkey patch 包装 model.model.model[-1].forward
       -> 补丁未生效(三档延迟几乎相同), 且 predict() 管线的预处理+NMS
          开销会淹没 decoder 的差异
    B. 顺序测各档位、每档只 20 样本
       -> 热漂移无法抵消

本脚本的做法:
    1. 直接改 DeformableTransformerDecoder.eval_idx(已验证有效: forward 里
       `elif i == self.eval_idx: ... break`), 调用 model(x) 原始前向
    2. 输入预先归一化到 0-1, 避开 ultralytics "Dividing input by 255" 告警
    3. 先充分预热到热稳态
    4. 各档位【交错】采样(round-robin): 每轮依次测 1..6 层, 重复 N 轮
       -> 热漂移被均摊到所有档位, 而非集中在某一档

用法:
    python verify_depth_timing.py --rounds 25 --runs 20
    python verify_depth_timing.py --depths 1 3 6 --rounds 40
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def log(m):
    print(m, flush=True)


def find_inner_decoder(model):
    """定位 DeformableTransformerDecoder(RTDETRDecoder.decoder)"""
    for mod in model.modules():
        if type(mod).__name__ == 'RTDETRDecoder' and hasattr(mod, 'decoder'):
            return mod.decoder
    return None


def gpu_temp():
    try:
        import subprocess
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu,clocks.current.sm',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except Exception:
        return 'n/a'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',
                    default=r'runs/detect/runs_rtdetr/r18_640_sub20_b4/weights/best.pt')
    ap.add_argument('--depths', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--rounds', type=int, default=25,
                    help='交错轮数; 每轮把所有档位各测一遍')
    ap.add_argument('--runs', type=int, default=20, help='每轮每档的前向次数')
    ap.add_argument('--warmup', type=int, default=60, help='初始预热次数(进热稳态)')
    ap.add_argument('--out', default='out_depth_timing.json')
    args = ap.parse_args()

    from ultralytics import RTDETR

    log(f'加载 {args.model}')
    m = RTDETR(args.model)
    model = m.model.cuda().eval()

    inner = find_inner_decoder(model)
    if inner is None:
        log('[错误] 找不到 DeformableTransformerDecoder')
        sys.exit(1)
    log(f'decoder: num_layers={inner.num_layers}  原 eval_idx={inner.eval_idx}')
    saved_idx = inner.eval_idx

    # 预归一化输入: 避免 ultralytics 对 0-255 范围输入打告警并再除 255
    x = torch.rand(1, 3, args.imgsz, args.imgsz, device='cuda')

    log(f'预热 {args.warmup} 次(进入热稳态)...  GPU: {gpu_temp()}')
    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
    torch.cuda.synchronize()
    log(f'预热完成。GPU: {gpu_temp()}')

    def measure(depth, n):
        """测一个档位 n 次, 返回毫秒列表"""
        inner.eval_idx = depth - 1
        with torch.no_grad():
            for _ in range(3):        # 切换档位后小预热, 让 kernel 选择稳定
                model(x)
        torch.cuda.synchronize()
        ts = []
        with torch.no_grad():
            for _ in range(n):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(x)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
        return ts

    # 交错采样: 每轮遍历所有档位, 让热漂移均摊
    samples = {d: [] for d in args.depths}
    log('')
    log(f'交错测量: {args.rounds} 轮 × {len(args.depths)} 档 × {args.runs} 次 '
        f'= 每档 {args.rounds * args.runs} 样本')
    t_start = time.time()
    for r in range(args.rounds):
        for d in args.depths:
            samples[d].extend(measure(d, args.runs))
        if (r + 1) % 5 == 0:
            log(f'  第 {r+1}/{args.rounds} 轮完成  '
                f'({time.time()-t_start:.0f}s)  GPU: {gpu_temp()}')

    inner.eval_idx = saved_idx

    # 统计
    stats = {}
    for d in args.depths:
        v = np.array(samples[d])
        stats[d] = dict(
            n=len(v),
            median=float(np.median(v)),
            mean=float(v.mean()),
            std=float(v.std()),
            p95=float(np.percentile(v, 95)),
            p99=float(np.percentile(v, 99)),
        )

    log('')
    log(f"{'层数':<6}{'样本':>7}{'中位(ms)':>11}{'均值(ms)':>11}{'标准差':>9}"
        f"{'P95':>9}{'P99':>9}")
    for d in args.depths:
        s = stats[d]
        log(f"{d:<6}{s['n']:>7}{s['median']:>11.2f}{s['mean']:>11.2f}"
            f"{s['std']:>9.2f}{s['p95']:>9.2f}{s['p99']:>9.2f}")

    full = max(args.depths)
    base = stats[full]['median']
    log('')
    log(f'=== 相对 {full} 层的省时 ===')
    inversions = []
    prev = None
    for d in args.depths:
        sav = (1 - stats[d]['median'] / base) * 100
        log(f'  {d} 层: {stats[d]["median"]:6.2f} ms   省时 {sav:+6.1f}%')
        if prev is not None and stats[d]['median'] < prev[1]:
            inversions.append((prev[0], d))
        prev = (d, stats[d]['median'])

    log('')
    if not inversions:
        log('结论: 延迟随层数【单调递增】, 符合预期。')
        log('      eval_depth_sweep.py 里"3 层比 6 层慢"无法复现,')
        log('      根因判定为热漂移+顺序效应(该脚本每档紧跟一次全量 val)。')
    else:
        log(f'结论: 仍存在逆序 {inversions}, 需进一步排查')
        log('      (kernel 选择 / 显存布局 / 该档位的 attention 分支差异)')

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(dict(model=args.model, imgsz=args.imgsz,
                       rounds=args.rounds, runs=args.runs,
                       interleaved=True,
                       stats={str(k): v for k, v in stats.items()}),
                  f, ensure_ascii=False, indent=2)
    log('')
    log(f'已写入 {args.out}')


if __name__ == '__main__':
    main()
