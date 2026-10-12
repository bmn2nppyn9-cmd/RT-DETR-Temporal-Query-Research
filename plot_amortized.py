# -*- coding: utf-8 -*-
"""
plot_amortized.py —— 关键帧调度下的摊薄延迟与吞吐
=================================================
单帧延迟只说明"某一帧能省多少", 真实收益取决于关键帧间隔 K:
    每 K 帧 1 个关键帧走完整 6 层, 其余 K-1 帧走截断的 d 层
    摊薄延迟 = (T_full + (K-1) * T_d) / K

注意: 本图是【调度层面的推算】, 不是端到端实测。
      T_full / T_d 取自 verify_depth_timing.py 的交错实测中位数。
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ('Microsoft YaHei', 'SimHei', 'DengXian'):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [f]
        break
plt.rcParams['axes.unicode_minus'] = False

lat = {int(k): v for k, v in
       json.load(open('out_depth_timing.json', encoding='utf-8'))['stats'].items()}
T_full = lat[6]['median']
Ks = list(range(1, 21))

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
colors = {1: '#B03030', 3: '#2E7D4F', 6: '#666666'}

# --- 图1: 摊薄延迟 vs K ---
ax = axes[0]
for d in (1, 3):
    y = [(T_full + (K - 1) * lat[d]['median']) / K for K in Ks]
    ax.plot(Ks, y, 'o-', ms=4, lw=2, color=colors[d],
            label=f'非关键帧用 {d} 层')
ax.axhline(T_full, color='#666', ls='--', lw=1.5, label='全程 6 层 (基线)')
ax.axhline(33.3, color='#B8860B', ls=':', lw=1.5)
ax.text(10.5, 34.2, '30 FPS 预算 33.3 ms', fontsize=8.5, color='#B8860B')
ax.set_xlabel('关键帧间隔 K (帧)')
ax.set_ylabel('摊薄单帧延迟 (ms)')
ax.set_title('摊薄延迟随 K 下降\n关键帧成本被 K-1 个轻帧摊薄', fontsize=11)
ax.set_xticks([1, 5, 10, 15, 20])
ax.grid(alpha=.25, ls=':')
ax.legend(fontsize=9)

# --- 图2: 等效吞吐 ---
ax = axes[1]
for d in (1, 3):
    y = [1000 / ((T_full + (K - 1) * lat[d]['median']) / K) for K in Ks]
    ax.plot(Ks, y, 'o-', ms=4, lw=2, color=colors[d], label=f'非关键帧用 {d} 层')
ax.axhline(1000 / T_full, color='#666', ls='--', lw=1.5,
           label=f'全程 6 层 = {1000/T_full:.0f} FPS')
ax.axhline(30, color='#B8860B', ls=':', lw=1.5)
ax.text(10.5, 31.5, '30 FPS', fontsize=8.5, color='#B8860B')
ax.set_xlabel('关键帧间隔 K (帧)')
ax.set_ylabel('摊薄吞吐 (FPS)')
ax.set_title('摊薄吞吐\nK=10 时 1 层可达 %.0f FPS' %
             (1000 / ((T_full + 9 * lat[1]['median']) / 10)), fontsize=11)
ax.set_xticks([1, 5, 10, 15, 20])
ax.grid(alpha=.25, ls=':')
ax.legend(fontsize=9, loc='lower right')

# --- 图3: 延迟分布的尾部收窄 ---
ax = axes[2]
depths = sorted(lat)
med = [lat[d]['median'] for d in depths]
p95 = [lat[d]['p95'] for d in depths]
p99 = [lat[d]['p99'] for d in depths]
ax.plot(depths, med, 'o-', color='#2D6FA8', lw=2, ms=6, label='中位')
ax.plot(depths, p95, 's--', color='#C4622D', lw=1.6, ms=5, label='P95')
ax.plot(depths, p99, '^:', color='#B03030', lw=1.6, ms=6, label='P99')
ax.fill_between(depths, med, p99, color='#B03030', alpha=.07)
for d in (1, 6):
    ax.annotate(f'P99-中位\n= {lat[d]["p99"]-lat[d]["median"]:.1f} ms',
                (d, lat[d]['p99']), textcoords='offset points',
                xytext=(6 if d == 1 else -52, 6), fontsize=8.5, color='#B03030')
ax.set_xlabel('Decoder 层数')
ax.set_ylabel('延迟 (ms)')
ax.set_title('尾部抖动随层数收窄\n截断同时改善 P99 有界性', fontsize=11)
ax.grid(alpha=.25, ls=':')
ax.legend(fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig('fig11_amortized.png', dpi=160, bbox_inches='tight')
plt.savefig('fig11_amortized.pdf', bbox_inches='tight')
print('已保存 fig11_amortized.png / .pdf')
print('')
print(f"{'K':>4}{'1层摊薄(ms)':>14}{'FPS':>8}{'3层摊薄(ms)':>14}{'FPS':>8}")
for K in (1, 2, 5, 10, 20):
    a = (T_full + (K - 1) * lat[1]['median']) / K
    b = (T_full + (K - 1) * lat[3]['median']) / K
    print(f'{K:>4}{a:>14.2f}{1000/a:>8.1f}{b:>14.2f}{1000/b:>8.1f}')
