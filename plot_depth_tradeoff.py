# -*- coding: utf-8 -*-
"""
plot_depth_tradeoff.py —— decoder 层数的精度-延迟权衡图
数据来源:
  延迟 out_depth_timing.json  —— verify_depth_timing.py 交错测量, 每档 500 样本,
                                 已排除热漂移(旧的 25.28ms 异常值不可复现)
  精度 out_depth_sweep_final.json —— eval_depth_sweep.py 全量 val 15062 张
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
depths = sorted(lat)
med = [lat[d]['median'] for d in depths]
p99 = [lat[d]['p99'] for d in depths]
std = [lat[d]['std'] for d in depths]

acc_d = [1, 3, 6]                       # 精度只在这三档跑了全量 val
map50 = [33.64, 38.65, 40.53]
map5095 = [19.53, 22.89, 23.93]
base_lat, base_map = lat[6]['median'], 40.53

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))

# --- 图1: 延迟 vs 层数(6 档全有) ---
ax = axes[0]
ax.errorbar(depths, med, yerr=std, fmt='o-', color='#C4622D', lw=2.2, ms=7,
            capsize=4, label='中位延迟 ±1σ', zorder=3)
ax.plot(depths, p99, 's--', color='#E0A579', lw=1.5, ms=5, label='P99', zorder=3)
for d, v in zip(depths, med):
    ax.annotate(f'{v:.1f}', (d, v), textcoords='offset points', xytext=(0, -16),
                ha='center', fontsize=8.5, color='#8A4520')
ax.set_xlabel('Decoder 层数')
ax.set_ylabel('单帧延迟 (ms)')
ax.set_title('延迟随层数单调递增\n每档 500 样本, 交错测量排除热漂移', fontsize=11)
ax.grid(alpha=.25, ls=':')
ax.legend(fontsize=9, loc='upper left')

# --- 图2: 精度 vs 层数 ---
ax = axes[1]
ax.plot(acc_d, map50, 'o-', color='#2D6FA8', lw=2.2, ms=8, label='mAP50')
ax.plot(acc_d, map5095, 's-', color='#7BA7CC', lw=2, ms=7, label='mAP50-95')
for d, v in zip(acc_d, map50):
    ax.annotate(f'{v:.1f}', (d, v), textcoords='offset points', xytext=(0, 9),
                ha='center', fontsize=9, color='#1E4E77')
ax.set_xlabel('Decoder 层数')
ax.set_ylabel('精度 (%)')
ax.set_title('精度随层数下降\n全量 val, 15062 张 / 253940 框', fontsize=11)
ax.set_xticks(acc_d)
ax.grid(alpha=.25, ls=':')
ax.legend(fontsize=9, loc='lower right')

# --- 图3: 权衡曲线 省时 vs 精度损失 ---
ax = axes[2]
sav = [(1 - lat[d]['median'] / base_lat) * 100 for d in acc_d]
dmap = [m - base_map for m in map50]
ax.plot(sav, dmap, 'o-', color='#3E7C4A', lw=2.2, ms=11, zorder=4)
ann = {1: (-4, -14), 3: (-52, 6), 6: (-46, -16)}
for s, dm, d in zip(sav, dmap, acc_d):
    ax.annotate(f'{d} 层\n省时 {s:.1f}%\nΔ {dm:+.2f} pp', (s, dm),
                textcoords='offset points', xytext=ann[d], fontsize=8.8,
                bbox=dict(boxstyle='round,pad=0.35', fc='#F2F6F2',
                          ec='#3E7C4A', alpha=.85))
ax.axhspan(-2, 0.4, color='#3E7C4A', alpha=.07, zorder=1)
ax.axhline(-2, color='#B03030', ls='--', lw=1.3, alpha=.75)
ax.text(0.5, -1.75, '预注册门槛 -2 pp', fontsize=8.5, color='#B03030')
ax.axhline(0, color='#888', lw=.8)
ax.set_xlabel('延迟节省 (%)')
ax.set_ylabel('mAP50 变化 (pp)')
ax.set_title('权衡曲线\n3 层: 省 21.2% 且落在门槛内', fontsize=11)
ax.set_xlim(-3, 43)
ax.grid(alpha=.25, ls=':')

plt.tight_layout()
plt.savefig('fig10_depth_tradeoff.png', dpi=160, bbox_inches='tight')
plt.savefig('fig10_depth_tradeoff.pdf', bbox_inches='tight')
print('已保存 fig10_depth_tradeoff.png / .pdf')
print('')
print(f"{'层数':<6}{'延迟(ms)':>11}{'省时':>9}{'mAP50':>9}{'ΔmAP50':>10}")
for d, m in zip(acc_d, map50):
    print(f"{d:<6}{lat[d]['median']:>11.2f}"
          f"{(1-lat[d]['median']/base_lat)*100:>8.1f}%{m:>9.2f}{m-base_map:>+10.2f}")
