# -*- coding: utf-8 -*-
"""
关键帧 / 非关键帧执行路径 —— 真实耗时实测
==========================================
把"理论节省上界"变成实测值。测量三条真实执行路径:

  路径A  FULL      : 关键帧 —— 完整前向 (Backbone + Encoder + Decoder)
  路径B  DEC_ONLY  : 非关键帧 —— 复用缓存的多尺度特征, 只跑 Decoder
  路径C  DEC_TRUNC : 非关键帧 —— 只跑 Decoder 且截断解码层数 (模拟自适应解码)

并按关键帧间隔 K 计算摊薄后的平均单帧耗时:
     avg(K) = [ FULL + (K-1) * NONKEY ] / K

实现要点:
  RT-DETR 的 RTDETRDecoder 输入为第 [21,24,27] 层的三个多尺度特征图。
  因此"只跑 decoder"= 缓存这三个张量后直接调用 decoder, 完全跳过 backbone+encoder。

用法:
  python bench_keyframe.py                        # FP16, 三档分辨率, K=1..10
  python bench_keyframe.py --fp32
  python bench_keyframe.py --res 608 960
  python bench_keyframe.py --dec-layers 1 3 6     # 指定截断层数
"""
import argparse
import time
import copy
import torch

BACKBONE_END = 10
ENCODER_END = 28
DEC_IDX = 28
DEC_INPUTS = [21, 24, 27]      # RTDETRDecoder 的输入来源层


def align32(v):
    return int(round(v / 32)) * 32


def build(scale, device, fp16):
    from ultralytics import RTDETR
    m = RTDETR(f'rtdetr-{scale}.yaml')
    model = m.model.to(device).eval()
    if fp16:
        model = model.half()
    return model


def assemble(layer, y, x):
    f = layer.f
    if isinstance(f, int):
        return y[f]
    if isinstance(f, list):
        return [y[j] for j in f]
    return y[-1] if y else x


@torch.no_grad()
def forward_to_encoder(inner, x):
    """跑到 encoder 结束, 返回 decoder 所需的三个多尺度特征(即缓存内容)"""
    y = []
    for i, layer in enumerate(inner):
        if i >= DEC_IDX:
            break
        x_in = x if i == 0 else assemble(layer, y, x)
        y.append(layer(x_in))
    return [y[j] for j in DEC_INPUTS]


@torch.no_grad()
def forward_full(inner, x):
    y = []
    for i, layer in enumerate(inner):
        x_in = x if i == 0 else assemble(layer, y, x)
        y.append(layer(x_in))
    return y[-1]


def timeit(fn, device, warmup=10, iters=50):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            if device == 'cuda':
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
            else:
                t0 = time.perf_counter()
                fn()
            ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


class TruncatedDecoder:
    """上下文管理器: 临时把 decoder 的有效解码层数截断为 n_layers, 退出时恢复。

    为什么不用 copy.copy: nn.Module 的浅拷贝会共享底层 _modules / __dict__,
    改副本会污染原对象(实测原 decoder 的 eval_idx 和 layers 会一起被改),
    导致后续测量全部失真。因此改为"原地临时修改 + 恢复"。

    截断依据: ultralytics 的 DeformableTransformerDecoder 在推理时靠
    `i == self.eval_idx` 触发 break 并收集输出, 所以只需把 eval_idx
    设为 n_layers-1, 即可让前向在第 n_layers 层后停止, 无需真的切 layers。
    """

    def __init__(self, dec, n_layers):
        self.d = dec.decoder
        self.n = n_layers
        self._saved = None

    def __enter__(self):
        self._saved = self.d.eval_idx
        self.d.eval_idx = self.n - 1
        return self

    def __exit__(self, *exc):
        self.d.eval_idx = self._saved
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', default='l')
    ap.add_argument('--res', nargs=2, type=int, default=None)
    ap.add_argument('--fp32', action='store_true', help='默认 FP16, 该开关切回 FP32')
    ap.add_argument('--dec-layers', nargs='*', type=int, default=[1, 3, 6])
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    fp16 = (not args.fp32) and device == 'cuda'
    raw = [tuple(args.res)] if args.res else [(600, 960), (800, 1280), (1200, 1920)]
    resolutions = [(align32(h), align32(w)) for h, w in raw]

    print('=' * 96)
    print(f'  关键帧 / 非关键帧执行路径实测    RT-DETR-{args.scale}    设备={device}    FP16={fp16}')
    if device == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print('=' * 96)

    model = build(args.scale, device, fp16)
    inner = model.model
    dec = inner[DEC_IDX]
    n_dec_layers = len(dec.decoder.layers)
    print(f'  Decoder 解码层数: {n_dec_layers}    查询数: {getattr(dec, "nq", "?")}\n')

    all_results = {}

    for res in resolutions:
        x = torch.randn(1, 3, res[0], res[1], device=device)
        if fp16:
            x = x.half()

        # 预先算出缓存特征(非关键帧复用它)
        feats = forward_to_encoder(inner, x)
        feats = [f.detach() for f in feats]

        # 路径A: 完整前向
        t_full = timeit(lambda: forward_full(inner, x), device)
        # 路径B: 只跑 decoder(复用缓存)
        t_dec = timeit(lambda: dec([f.clone() for f in feats]), device)
        # 路径B': 纯 decoder(不含 clone 开销)作为对照
        t_dec_pure = timeit(lambda: dec(feats), device)

        # 路径C: 截断解码层数
        trunc = {}
        for nl in args.dec_layers:
            if nl > n_dec_layers:
                continue
            with TruncatedDecoder(dec, nl):
                trunc[nl] = timeit(lambda: dec(feats), device)

        print(f'  ── 分辨率 {res[0]}×{res[1]} ' + '─' * 60)
        print(f'     路径A  FULL       (Backbone+Encoder+Decoder) : {t_full:7.2f} ms')
        print(f'     路径B  DEC_ONLY   (复用缓存, 只跑 Decoder)   : {t_dec_pure:7.2f} ms'
              f'   → 单帧节省 {(1-t_dec_pure/t_full)*100:5.1f}%')
        print(f'            (含特征拷贝开销的保守值)              : {t_dec:7.2f} ms')
        for nl, tv in sorted(trunc.items()):
            print(f'     路径C  DEC_{nl}层    (只跑 Decoder 前 {nl} 层)      : {tv:7.2f} ms'
                  f'   → 单帧节省 {(1-tv/t_full)*100:5.1f}%')

        # 摊薄平均耗时
        print(f'\n     关键帧间隔 K 下的平均单帧耗时 (摊薄):')
        hdr = f'       {"K":>3} | {"仅查询传播":>12} | ' + ' | '.join(
            f'{"传播+"+str(nl)+"层":>12}' for nl in sorted(trunc)) + f' | {"逐帧全算":>10}'
        print(hdr)
        print('       ' + '-' * (len(hdr) - 7))
        for K in [1, 2, 3, 5, 8, 10]:
            row = f'       {K:>3} | '
            avg_b = (t_full + (K - 1) * t_dec_pure) / K
            row += f'{avg_b:9.2f}ms | '
            cells = []
            for nl in sorted(trunc):
                avg_c = (t_full + (K - 1) * trunc[nl]) / K
                cells.append(f'{avg_c:9.2f}ms')
            row += ' | '.join(cells)
            row += f' | {t_full:7.2f}ms'
            print(row)
        print()

        all_results[res] = dict(full=t_full, dec=t_dec_pure, dec_copy=t_dec, trunc=trunc)

    # ── 汇总 ──
    print('=' * 96)
    print('  【汇总:非关键帧相对完整前向的节省比例】')
    print(f'  {"分辨率":>12} | {"只跑Decoder":>13} | ' + ' | '.join(
        f'{"Decoder前"+str(nl)+"层":>14}' for nl in args.dec_layers if nl <= n_dec_layers))
    print('  ' + '-' * 78)
    for res, r in all_results.items():
        line = f'  {res[0]}×{res[1]:<7} | {(1-r["dec"]/r["full"])*100:11.1f}% | '
        line += ' | '.join(f'{(1-r["trunc"][nl]/r["full"])*100:12.1f}%'
                           for nl in sorted(r['trunc']))
        print(line)
    print('=' * 96)
    print("""
  说明:
   · "只跑 Decoder" 是查询传播的核心收益 —— 非关键帧完全跳过 Backbone+Encoder。
   · "Decoder 前 N 层" 叠加了自适应解码的收益。
   · 摊薄表给出关键帧间隔 K 下的等效平均耗时, 可直接换算等效 FPS。
   · 未计入查询传播模块与门控自身的开销, 属结构性上界的实测近似。
""")


if __name__ == '__main__':
    main()
