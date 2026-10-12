# -*- coding: utf-8 -*-
"""
RT-DETR 端侧部署可行性分析 —— 模块耗时拆解脚本
==================================================
回答三个问题:
  Q1. RT-DETR 的 decoder 占总耗时多少?       (决定"自适应解码"能否站住)
  Q2. 耗时随输入分辨率如何变化?              (决定"查询传播解耦分辨率"论点能否站住)
  Q3. 参数量 / 显存分布如何?

用法:
  python profile_rtdetr.py                 # rtdetr-l, 三档分辨率
  python profile_rtdetr.py --scale x
  python profile_rtdetr.py --res 600 960
  python profile_rtdetr.py --fp16

说明:
  - 随机初始化权重即可。耗时占比由结构决定, 与权重数值无关。
  - 结果用于"相对分析", 不能直接当作 RK3588 的绝对延迟。
"""
import argparse
import time
import torch

# --- 模块切分: 按层类型动态定位, 兼容 HGNet 版(l/x) 与 ResNet 版(resnet50/101) ---
#   Backbone: 开头的 HGStem/HGBlock/ResNetLayer 连续段
#   Encoder:  backbone 之后、RTDETRDecoder 之前(AIFI + CCFM / 特征融合层)
#   Decoder:  RTDETRDecoder (deformable attention, 端侧最贵)
BACKBONE_TYPES = {'HGStem', 'HGBlock', 'ResNetLayer'}
SEGMENTS = ['Backbone', 'Encoder', 'Decoder']


def layer_segments(inner):
    """返回 (每层所属分段列表, (backbone结束, decoder下标))"""
    names = [type(layer).__name__ for layer in inner]
    bb_end = 0
    for name in names:
        if name in BACKBONE_TYPES:
            bb_end += 1
        else:
            break
    dec_idx = names.index('RTDETRDecoder')
    segs = ['Backbone' if i < bb_end else 'Encoder' if i < dec_idx else 'Decoder'
            for i in range(len(inner))]
    return segs, (bb_end, dec_idx)


def build_model(scale='l', device='cuda'):
    from ultralytics import RTDETR
    m = RTDETR(f'rtdetr-{scale}.yaml')
    model = m.model.to(device)
    model.eval()
    return model


def assemble_input(layer, y, x):
    """复现 ultralytics 的前向输入组装逻辑"""
    f = layer.f
    if isinstance(f, int):
        return y[f]
    if isinstance(f, list):
        return [y[j] for j in f]
    return y[-1] if y else x


def timed_forward(inner, x, device, layer_segs, measure=True):
    """完整前向, 逐层计时。

    关键: CUDA 是异步的。若只在层间打 event 而不同步, GPU 实际执行会被推迟,
    等待时间会全部堆积到最后一层, 导致 Decoder 被严重高估。
    因此测量模式下每层后强制 synchronize, 以获得正确的逐层归因。
    (代价: 串行化带来固定开销, 但对"占比"这一相对量的影响可接受)
    """
    y = []
    stamps = {s: 0.0 for s in SEGMENTS}
    for i, layer in enumerate(inner):
        seg = layer_segs[i]
        x_in = x if i == 0 else assemble_input(layer, y, x)
        if measure and device == 'cuda':
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = layer(x_in)
            torch.cuda.synchronize()
            stamps[seg] += (time.perf_counter() - t0) * 1000
        elif measure:
            t0 = time.perf_counter()
            out = layer(x_in)
            stamps[seg] += (time.perf_counter() - t0) * 1000
        else:
            out = layer(x_in)
        y.append(out)
    return y, stamps


def median(vals):
    vals = sorted(vals)
    return vals[len(vals) // 2] if vals else 0.0


def align32(v):
    """RT-DETR 要求输入边长为 32 的倍数, 否则 FPN 上采样拼接会差 1 像素"""
    return int(round(v / 32)) * 32


def profile_segments(model, res, device, fp16, layer_segs, warmup=8, iters=30):
    inner = model.model
    x = torch.randn(1, 3, res[0], res[1], device=device)
    if fp16:
        inner = inner.half()
        x = x.half()

    # warmup(不计时, 走真实异步路径)
    with torch.no_grad():
        for _ in range(warmup):
            timed_forward(inner, x, device, layer_segs, measure=False)
    if device == 'cuda':
        torch.cuda.synchronize()

    # 1) 分段计时(每层同步, 用于归因占比)
    acc = {s: [] for s in SEGMENTS}
    with torch.no_grad():
        for _ in range(iters):
            _, stamps = timed_forward(inner, x, device, layer_segs, measure=True)
            for seg in SEGMENTS:
                acc[seg].append(stamps[seg])
    seg_med = {s: median(acc[s]) for s in SEGMENTS}

    # 2) 端到端计时(不插同步, 反映真实推理延迟) —— 用于交叉校验
    e2e = []
    with torch.no_grad():
        for _ in range(iters):
            if device == 'cuda':
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                timed_forward(inner, x, device, layer_segs, measure=False)
                torch.cuda.synchronize()
            else:
                t0 = time.perf_counter()
                timed_forward(inner, x, device, layer_segs, measure=False)
            e2e.append((time.perf_counter() - t0) * 1000)

    return seg_med, median(e2e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', default='l', choices=['l', 'x', 'resnet50', 'resnet101'])
    ap.add_argument('--res', nargs=2, type=int, default=None)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--cpu', action='store_true', help='用 CPU 测(仅参考)')
    args = ap.parse_args()

    device = 'cpu' if args.cpu else 'cuda'
    if device == 'cuda' and not torch.cuda.is_available():
        print('[!] CUDA 不可用, 自动切到 CPU')
        device = 'cpu'

    raw_res = [tuple(args.res)] if args.res else [(600, 960), (800, 1280), (1200, 1920)]
    resolutions = [(align32(h), align32(w)) for h, w in raw_res]
    for (rh, rw), (h, w) in zip(resolutions, raw_res):
        if (rh, rw) != (h, w):
            print(f'[i] 分辨率 {h}x{w} 已对齐到 32 的倍数 -> {rh}x{rw}')
    use_fp16 = args.fp16 and device == 'cuda'

    print('=' * 78)
    print(f'  RT-DETR ({args.scale}) 模块耗时拆解    设备={device}    FP16={use_fp16}')
    if device == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print('=' * 78)

    model = build_model(args.scale, device)
    layer_segs, (bb_end, dec_idx) = layer_segments(model.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  参数量: {n_params/1e6:.2f} M')
    print(f'  层切分: Backbone [0..{bb_end-1}]  Encoder [{bb_end}..{dec_idx-1}]  Decoder [{dec_idx}]\n')

    print(f'{"分辨率":>11} | {"Backbone":>16} | {"Encoder":>16} | {"Decoder":>16} | {"分段和":>8} | {"端到端":>8} | {"显存":>7}')
    print('-' * 108)

    summary = []
    for res in resolutions:
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            seg, e2e = profile_segments(model, res, device, use_fp16, layer_segs)
        except Exception as e:
            print(f'  {res} 测量失败: {type(e).__name__}: {e}')
            continue
        total = sum(seg.values())
        line = f'{res[0]}x{res[1]:<6} | '
        line += ' | '.join(f'{seg[s]:>7.2f}ms{seg[s]/total*100:>6.1f}%' for s in SEGMENTS)
        line += f' | {total:>6.2f}ms | {e2e:>6.2f}ms |'
        if device == 'cuda':
            line += f' {torch.cuda.max_memory_allocated()/1024**3:>5.2f}GB'
        print(line)
        summary.append((res, seg, total, e2e))

    print('=' * 78)
    print('  【关键结论】')
    if len(summary) >= 2:
        f0, f1 = summary[0], summary[-1]
        for s in SEGMENTS:
            print(f'    {s:>9}: 占比 {f0[1][s]/f0[2]*100:5.1f}% ({f0[1][s]:6.2f}ms) '
                  f'-> {f1[1][s]/f1[2]*100:5.1f}% ({f1[1][s]:6.2f}ms)   '
                  f'增长 {f1[1][s]/max(f0[1][s],1e-9):5.2f}x')
        print(f'    {"分段和":>9}: {f0[2]:6.2f}ms -> {f1[2]:6.2f}ms   增长 {f1[2]/f0[2]:5.2f}x')
        print(f'    {"端到端":>9}: {f0[3]:6.2f}ms -> {f1[3]:6.2f}ms   增长 {f1[3]/f0[3]:5.2f}x')
    elif len(summary) == 1:
        r, seg, total, e2e = summary[0]
        for s in SEGMENTS:
            print(f'    {s:>9}: {seg[s]:6.2f}ms  ({seg[s]/total*100:5.1f}%)')
        print(f'    {"端到端":>9}: {e2e:6.2f}ms')
    print('=' * 78)
    print("""
  解读:
    1) Decoder 占比显著(>25%) -> "自适应解码/查询传播"节省的是关键路径, P3 成立。
    2) Decoder 占比随分辨率上升而增长 -> 它比 backbone 更吃高分辨率, 解耦论点更强。
    3) Decoder 占比很小(<10%) -> 省它意义有限, P3 需重新论证。
""")


if __name__ == '__main__':
    main()
