# -*- coding: utf-8 -*-
"""
四段拆解与非关键帧真实路径实测
================================
回答一个问题: 非关键帧到底跳掉了什么、保留了什么, 各值多少 ms。

四段切分(按层索引, 以 rtdetr-l.yaml 为准):
  段1 SHALLOW  层 0-3   浅段 Backbone (HGStem+stage1@stride4 + stage2@stride8)
                        -> 非关键帧唯一新像素入口, 残差通道的特征来源
  段2 DEEP     层 4-9   深层 Backbone (stage3@stride16, 18块 + stage4@stride32)
                        -> 非关键帧跳过
  段3 ENCODER  层 10-27 AIFI + CCFM 双向融合 -> [21,24,27] 三尺度 256ch
                        -> 非关键帧跳过, 读关键帧缓存 = Decoder 的 KV
  段4 DECODER  层 28    RTDETRDecoder (enc_head + 6层 deformable + head)
                        -> 非关键帧截断跑 1/3/6 层

实测路径(端到端, 不插同步, median + P99):
  FULL  完整前向 (关键帧)
  B     缓存KV + 完整6层Decoder        (零新像素输入, 原"上界"口径)
  N6    浅段+残差门控 + 6层Decoder     (修正后非关键帧路径)
  N3    浅段+残差门控 + 3层Decoder
  N1    浅段+残差门控 + 1层Decoder

残差门控为随机初始化代理模块, 只用于耗时, 不代表精度。
另: 打印缓存张量真实大小; 首尾复测 FULL@608 检测降频漂移。
"""
import argparse
import time
import torch
import torch.nn as nn

SHALLOW_END = 4          # 层 0..3
DEEP_END = 10            # 层 4..9
DEC_IDX = 28             # RTDETRDecoder
DEC_INPUTS = [21, 24, 27]

SEG_NAMES = ['SHALLOW(0-3)', 'DEEP(4-9)', 'ENCODER(10-27)', 'DECODER(28)']


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


def seg_of(i):
    if i < SHALLOW_END:
        return 0
    if i < DEEP_END:
        return 1
    if i < DEC_IDX:
        return 2
    return 3


@torch.no_grad()
def forward_layers(inner, x, lo, hi, y=None):
    """只跑层 [lo, hi), 返回逐层输出列表 y(若给则在其上续跑)"""
    if y is None:
        y = []
    for i in range(lo, hi):
        layer = inner[i]
        x_in = x if i == 0 else assemble(layer, y, x)
        y.append(layer(x_in))
    return y


class ResidualGate(nn.Module):
    """残差+门控代理: 浅特征降维 -> 与关键帧基准做差 -> mask头 + 门控MLP"""
    def __init__(self, cin=512, cmid=64):
        super().__init__()
        self.proj = nn.Conv2d(cin, cmid, 1, bias=False)
        self.mask_head = nn.Sequential(
            nn.Conv2d(cmid, cmid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(cmid, 1, 1))
        self.gate = nn.Sequential(
            nn.Linear(cmid, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, cur, ref):
        d = self.proj(cur) - ref
        mask = self.mask_head(d)
        g = self.gate(d.abs().mean(dim=(2, 3)))
        return mask, g


class MiniGate(nn.Module):
    """独立微型变化头: 不碰 backbone, 图像域 stride-8 灰度差分 + 小CNN。
    缓存基准仅 1ch 灰度图(约 72KB @1216), 与共享浅段方案(4.45MB)差 60 倍。"""
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(2, 16, 3, 2, 1), nn.ReLU(),    # -> stride 16
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),   # -> stride 32
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.mask_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 1, 1))
        self.gate = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, cur_gray8, ref_gray8):
        d = torch.cat([cur_gray8, cur_gray8 - ref_gray8], 1)
        f = self.body(d)
        mask = self.mask_head(f)
        g = self.gate(f.mean(dim=(2, 3)))
        return mask, g


class TruncatedDecoder:
    """临时截断 decoder 有效层数(原地改 eval_idx, 退出恢复)"""
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


def timeit(fn, device, warmup=20, iters=50):
    """返回 (median_ms, p99_ms)"""
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
    return ts[len(ts) // 2], ts[int(len(ts) * 0.99) - 1]


def mib(t):
    return t.numel() * t.element_size() / 1024 ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', default='l')
    ap.add_argument('--fp32', action='store_true')
    ap.add_argument('--iters', type=int, default=50)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16 = (not args.fp32) and device == 'cuda'
    resolutions = [(align32(h), align32(w)) for h, w in
                   [(608, 960), (800, 1280), (1216, 1920)]]

    print('=' * 100)
    print(f'  四段拆解 + 非关键帧真实路径实测   RT-DETR-{args.scale}  设备={device}  FP16={fp16}')
    if device == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print('=' * 100)

    model = build(args.scale, device, fp16)
    inner = model.model
    dec = inner[DEC_IDX]

    # ── 0. 层清单核对 ──
    print('\n【层清单核对】(确认四段切点)')
    for i, layer in enumerate(inner):
        mark = ['◆ 浅段', '◇ 深层', '□ Encoder', '■ Decoder'][seg_of(i)]
        print(f'  层{i:>2}  {type(layer).__name__:>22}  {mark}')
    n_dec = len(dec.decoder.layers)
    print(f'  Decoder 层数: {n_dec}')

    # 残差门控代理模块(与浅段输出同精度同设备)
    probe = ResidualGate(cin=512, cmid=64).to(device).eval()
    mini = MiniGate().to(device).eval()
    if fp16:
        probe = probe.half()
        mini = mini.half()

    # ── 降频漂移检测: 先测一次 FULL@608, 结尾复测 ──
    drift = {}

    results = {}
    for ri, res in enumerate(resolutions):
        x = torch.randn(1, 3, res[0], res[1], device=device)
        if fp16:
            x = x.half()

        # ── 1. 逐层同步计时 -> 四段归因 (每迭代先求段和, 再取 median) ──
        y = forward_layers(inner, x, 0, DEC_IDX + 1)   # warmup 全程一次
        stamps = {s: [] for s in range(4)}
        with torch.no_grad():
            for _ in range(args.iters):
                yy = []
                seg_acc = {s: 0.0 for s in range(4)}
                for i, layer in enumerate(inner):
                    x_in = x if i == 0 else assemble(layer, yy, x)
                    torch.cuda.synchronize() if device == 'cuda' else None
                    t0 = time.perf_counter()
                    out = layer(x_in)
                    torch.cuda.synchronize() if device == 'cuda' else None
                    seg_acc[seg_of(i)] += (time.perf_counter() - t0) * 1000
                    yy.append(out)
                for s in range(4):
                    stamps[s].append(seg_acc[s])
        seg_med = {s: sorted(stamps[s])[len(stamps[s]) // 2] for s in range(4)}
        seg_sum = sum(seg_med.values())

        # ── 2. 缓存内容准备 ──
        y_shallow = forward_layers(inner, x, 0, SHALLOW_END)
        feat_s8 = y_shallow[3]                              # 浅段输出 (残差通道输入)
        y_enc = forward_layers(inner, x, SHALLOW_END, DEC_IDX, list(y_shallow))
        kv = [y_enc[j].detach() for j in DEC_INPUTS]        # KV 缓存
        with torch.no_grad():
            ref = probe.proj(feat_s8).detach()              # 残差基准(降维后缓存)

        # ── 3. 端到端路径实测 (median, p99) ──
        gray8 = torch.nn.functional.avg_pool2d(x.float().mean(1, keepdim=True), 8)
        if fp16:
            gray8 = gray8.half()
        ref_gray8 = gray8.clone()                     # 关键帧灰度基准(模拟缓存)

        def run_full():
            forward_layers(inner, x, 0, DEC_IDX + 1)

        def run_shallow_res():
            yy = forward_layers(inner, x, 0, SHALLOW_END)
            probe(yy[3], ref)

        paths = {}
        paths['FULL'] = timeit(run_full, device, iters=args.iters)
        paths['B(dec6)'] = timeit(lambda: dec(kv), device, iters=args.iters)
        t_shres = timeit(run_shallow_res, device, iters=args.iters)
        t_probe_only = timeit(lambda: probe(feat_s8, ref), device, iters=args.iters)
        t_mini_only = timeit(lambda: mini(gray8, ref_gray8), device, iters=args.iters)
        dec_t = {}
        for nl in [6, 3, 1]:
            with TruncatedDecoder(dec, nl):
                dec_t[nl] = timeit(lambda: dec(kv), device, iters=args.iters)
            # N 路径: 共享浅段 + 残差门控 + 截断 decoder
            def run_n(nl_=nl):
                yy = forward_layers(inner, x, 0, SHALLOW_END)
                probe(yy[3], ref)
                with TruncatedDecoder(dec, nl_):
                    dec(kv)
            paths[f'N{nl}'] = timeit(run_n, device, iters=args.iters)
            # M 路径: 独立微头 + 截断 decoder (不碰 backbone)
            def run_m(nl_=nl):
                mini(gray8, ref_gray8)
                with TruncatedDecoder(dec, nl_):
                    dec(kv)
            paths[f'M{nl}'] = timeit(run_m, device, iters=args.iters)
        paths['浅段+残差(单独)'] = t_shres

        if ri == 0:
            drift['first'] = paths['FULL'][0]

        # ── 4. 打印 ──
        print(f'\n{"─" * 100}')
        print(f'  分辨率 {res[0]}×{res[1]}')
        print(f'  四段耗时(逐层同步, median):')
        for s in range(4):
            print(f'    {SEG_NAMES[s]:<16}: {seg_med[s]:7.2f} ms  ({seg_med[s] / seg_sum * 100:5.1f}%)')
        print(f'    {"分段和":<16}: {seg_sum:7.2f} ms')
        print(f'  缓存大小: KV[21,24,27] = {sum(mib(t) for t in kv):6.2f} MiB '
              f'(三尺度 256ch)   残差基准(proj后64ch) = {mib(ref):5.2f} MiB   '
              f'浅特征原始(512ch) = {mib(feat_s8):6.2f} MiB   微头灰度基准 = {mib(ref_gray8):5.3f} MiB')
        print(f'  Decoder 截断单独耗时 (median / P99): '
              + '   '.join(f'dec{nl}={dec_t[nl][0]:5.2f}/{dec_t[nl][1]:5.2f}ms' for nl in [6, 3, 1]))
        print(f'  残差门控单独: 共享浅段版probe={t_probe_only[0]:5.2f}ms   独立微头mini={t_mini_only[0]:5.2f}ms')
        print(f'  端到端路径 (median / P99):')
        full_med = paths['FULL'][0]
        for k in ['FULL', 'B(dec6)', 'N6', 'N3', 'N1', 'M6', 'M3', 'M1']:
            med, p99 = paths[k]
            print(f'    {k:<10}: {med:7.2f} / {p99:7.2f} ms   占FULL {med / full_med * 100:5.1f}%   '
                  f'节省 {(1 - med / full_med) * 100:5.1f}%   等效FPS {1000 / med:6.1f}')
        print(f'    浅段+残差门控单独: {t_shres[0]:6.2f} / {t_shres[1]:6.2f} ms '
              f'(占FULL {t_shres[0] / full_med * 100:4.1f}%)')

        results[res] = dict(seg=seg_med, seg_sum=seg_sum, paths=paths,
                            kv_mib=sum(mib(t) for t in kv), ref_mib=mib(ref))

    # ── 5. 降频漂移复测 ──
    x0 = torch.randn(1, 3, resolutions[0][0], resolutions[0][1], device=device)
    if fp16:
        x0 = x0.half()
    t_recheck, _ = timeit(lambda: forward_layers(inner, x0, 0, DEC_IDX + 1), device, iters=30)
    print(f'\n{"─" * 100}')
    print(f'  降频漂移检查: FULL@{resolutions[0][0]}×{resolutions[0][1]} '
          f'首测 {drift["first"]:.2f} ms -> 末测 {t_recheck:.2f} ms '
          f'(漂移 {(t_recheck / drift["first"] - 1) * 100:+.1f}%)')

    # ── 6. 汇总 ──
    print(f'\n{"=" * 100}')
    print('  【汇总: 修正前后非关键帧节省对比 (median)】')
    hdr = (f'  {"分辨率":>11} | {"FULL":>8} | {"B旧上界":>13} | {"N6":>13} | {"N3":>13} | {"N1":>13} |'
           f' {"M6":>13} | {"M3":>13} | {"M1":>13}')
    print(hdr)
    for res, r in results.items():
        f = r['paths']['FULL'][0]
        row = f'  {res[0]}×{res[1]:<6} | {f:6.2f}ms |'
        for k in ['B(dec6)', 'N6', 'N3', 'N1', 'M6', 'M3', 'M1']:
            med = r['paths'][k][0]
            row += f' {med:6.2f}/-{(1 - med / f) * 100:4.1f}% |'
        print(row)
    print(f'{"=" * 100}')

    # ── 7. 画图 fig9 ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2))
        ress = list(results.keys())
        labels = [f'{r[0]}×{r[1]}' for r in ress]
        colors = ['#2E5C9A', '#C0392B', '#2E7D4F', '#B8860B']
        bottoms = np.zeros(len(ress))
        for s in range(4):
            vals = [results[r]['seg'][s] for r in ress]
            ax1.bar(labels, vals, bottom=bottoms, color=colors[s], label=SEG_NAMES[s])
            for i, v in enumerate(vals):
                if v > 2:
                    ax1.text(i, bottoms[i] + v / 2, f'{v:.1f}', ha='center', va='center',
                             fontsize=9, color='white')
            bottoms += np.array(vals)
        for i, r in enumerate(ress):
            ax1.text(i, bottoms[i] + 1, f'{bottoms[i]:.1f}ms', ha='center', fontsize=10)
        ax1.set_ylabel('耗时 (ms, 逐层同步 median)')
        ax1.set_title('四段拆解: 浅段/深层/Encoder/Decoder')
        ax1.legend(fontsize=9)
        ax1.set_ylim(0, bottoms.max() * 1.18)

        keys = ['FULL', 'B(dec6)', 'N6', 'N1', 'M6', 'M1']
        klabels = ['逐帧全算', 'B 旧上界\n(0像素+6层)', 'N6 浅段残差\n+6层', 'N1 浅段残差\n+1层',
                   'M6 微头\n+6层', 'M1 微头\n+1层']
        xpos = np.arange(len(keys))
        wdt = 0.26
        for j, r in enumerate(ress):
            f = results[r]['paths']['FULL'][0]
            vals = [results[r]['paths'][k][0] / f * 100 for k in keys]
            ax2.bar(xpos + (j - 1) * wdt, vals, wdt, label=labels[j], color=colors[j])
            for i, v in enumerate(vals):
                ax2.text(xpos[i] + (j - 1) * wdt, v + 1.5, f'{v:.0f}', ha='center',
                         fontsize=8, color=colors[j])
        ax2.set_xticks(xpos)
        ax2.set_xticklabels(klabels, fontsize=9)
        ax2.set_ylabel('相对 FULL 的耗时 (%)')
        ax2.set_title('非关键帧路径修正: 旧上界 vs 浅段+残差真实路径')
        ax2.axhline(100, color='gray', ls='--', lw=0.8)
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, 118)

        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(f'fig9_stage_split.{ext}', dpi=200 if ext == 'png' else None)
        print('  图已保存: fig9_stage_split.png / .pdf')
    except Exception as e:
        print(f'  画图失败(不影响数据): {e}')


if __name__ == '__main__':
    main()
