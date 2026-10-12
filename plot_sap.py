# -*- coding: utf-8 -*-
"""
plot_sap.py —— sAP vs 端到端延迟曲线(流式鲁棒性)
================================================
数据: out_sap.json (eval_sap.py --sweep, 已修正 GT 分母: 丢帧计为漏检)
含义: 每条曲线是一个固定解码深度。曲线在"悬崖"处崩塌 —— 那是
      推理耗时 + 注入延迟 越过帧间隔预算的时刻。
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

d = json.load(open('out_sap.json', encoding='utf-8'))
sweep, res = d['sweep'], d['results']
colors = {'1': '#B03030', '3': '#2E7D4F', '6': '#2D6FA8'}
marks = {'1': 'o', '3': 's', '6': '^'}

# 各深度实测推理耗时(录制所得中位数), 用于标注悬崖预测位置
COMPUTE = {1: 13.18, 3: 16.10, 6: 19.89}
BUDGET = 33.3

fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))

# --- 图1: sAP50 vs 注入延迟 ---
ax = axes[0]
for k in sorted(res, key=int):
    y = [r['sap50'] for r in res[k]]
    ax.plot(sweep, y, marks[k] + '-', color=colors[k], lw=2.2, ms=7,
            label=f'{k} 层解码')
for k in sorted(res, key=int):
    cliff = BUDGET - COMPUTE[int(k)]
    ax.axvline(cliff, color=colors[k], ls=':', lw=1.4, alpha=.65)
    ax.annotate(f'{k}层悬崖\n{cliff:.1f}ms', (cliff, 0.20),
                textcoords='offset points', xytext=(3, 0), fontsize=8,
                color=colors[k], rotation=90, va='bottom')
ax.set_xlabel('注入延迟 (ms)')
ax.set_ylabel('sAP50')
ax.set_title('sAP 随延迟的崩塌\n越深的解码器越早跌下悬崖', fontsize=12)
ax.grid(alpha=.25, ls=':')
ax.legend(fontsize=10, loc='lower left')
ax.set_ylim(0.10, 0.42)

# --- 图2: 丢弃率 ---
ax = axes[1]
for k in sorted(res, key=int):
    y = [r['drop_pct'] for r in res[k]]
    ax.plot(sweep, y, marks[k] + '-', color=colors[k], lw=2.2, ms=7,
            label=f'{k} 层解码')
ax.set_xlabel('注入延迟 (ms)')
ax.set_ylabel('丢弃帧比例 (%)')
ax.set_title('帧丢弃率\n悬崖处丢弃率骤升 —— 处理器跟不上采样率', fontsize=12)
ax.grid(alpha=.25, ls=':')
ax.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig('fig12_sap_latency.png', dpi=160, bbox_inches='tight')
plt.savefig('fig12_sap_latency.pdf', bbox_inches='tight')
print('已保存 fig12_sap_latency.png / .pdf')
print('')
print('=== 关键对比: 各延迟下的最优深度 ===')
hdr = f"{'注入L':>6}" + ''.join(f'{k+"层":>11}' for k in sorted(res, key=int)) + '   最优'
print(hdr)
for i, L in enumerate(sweep):
    vals = {k: res[k][i]['sap50'] for k in res}
    best = max(vals, key=vals.get)
    row = f'{L:>6.0f}' + ''.join(f'{vals[k]:>11.4f}' for k in sorted(res, key=int))
    print(row + f'   {best} 层')
print('')
print(f"预算 {BUDGET}ms, 各深度推理耗时 " +
      ', '.join(f'{k}层={v}ms' for k, v in sorted(COMPUTE.items())))
