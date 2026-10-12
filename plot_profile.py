# -*- coding: utf-8 -*-
"""
RT-DETR 模块耗时分析 —— 绘图脚本
输出 4 张图(PNG 300dpi + PDF 矢量),可直接放入 PPT / 论文:
  fig1_breakdown_stacked.png   模块耗时堆叠图(FP32 vs FP16)
  fig2_scaling_curve.png       各模块耗时随分辨率增长曲线(核心发现)
  fig3_ratio_shift.png         模块占比随分辨率的变化
  fig4_savings_potential.png   不同策略的理论节省潜力
数据来源:5070Ti Laptop 实测(profile_rtdetr.py)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 300

# ============================ 实测数据 ============================
RES_LABELS = ['608×960', '800×1280', '1216×1920']
PIXELS = np.array([608*960, 800*1280, 1216*1920], dtype=float)
PIX_REL = PIXELS / PIXELS[0]

# FP32(单位 ms)
FP32 = {
    'Backbone': [13.07, 19.10, 41.24],
    'Encoder':  [10.83, 14.38, 30.26],
    'Decoder':  [12.06, 12.93, 15.96],
}
FP32_E2E = [31.33, 44.35, 75.01]

# FP16(单位 ms)
FP16 = {
    'Backbone': [8.65, 10.33, 19.99],
    'Encoder':  [6.50,  7.35, 12.93],
    'Decoder':  [11.54, 12.06, 12.56],
}
FP16_E2E = [27.20, 28.62, 44.61]

SEGS = ['Backbone', 'Encoder', 'Decoder']
COLORS = {'Backbone': '#4C72B0', 'Encoder': '#55A868', 'Decoder': '#C44E52'}

TITLE_NOTE = 'RT-DETR-L @ RTX 5070 Ti Laptop（逐层同步计时，中位数）'


# ============================ 图1:堆叠柱状 ============================
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (data, e2e, tag) in zip(axes, [(FP32, FP32_E2E, 'FP32'), (FP16, FP16_E2E, 'FP16')]):
        x = np.arange(len(RES_LABELS))
        bottom = np.zeros(len(RES_LABELS))
        for s in SEGS:
            v = np.array(data[s])
            bars = ax.bar(x, v, 0.55, bottom=bottom, label=s,
                          color=COLORS[s], edgecolor='white', linewidth=0.8)
            for xi, (vi, bi) in enumerate(zip(v, bottom)):
                tot = sum(data[k][xi] for k in SEGS)
                if vi / tot > 0.10:
                    ax.text(xi, bi + vi / 2, f'{vi:.1f}\n{vi/tot*100:.0f}%',
                            ha='center', va='center', fontsize=9,
                            color='white', fontweight='bold')
            bottom += v
        # 端到端参考线(放到柱子后面, 避免压住数值标签)
        ax.plot(x, e2e, 'o--', color='#222222', lw=1.4, ms=5,
                label='端到端实测', zorder=0, alpha=0.75)
        ax.set_xticks(x); ax.set_xticklabels(RES_LABELS)
        ax.set_title(f'{tag}', fontweight='bold')
        ax.set_xlabel('输入分辨率')
        ax.grid(axis='y', alpha=0.3, ls=':')
        ax.set_axisbelow(True)
    axes[0].set_ylabel('单帧耗时 (ms)')
    axes[0].legend(loc='upper left', framealpha=0.95, fontsize=9)
    fig.suptitle('图1  RT-DETR 各模块耗时分解', fontsize=13, fontweight='bold', y=1.02)
    fig.text(0.5, -0.06, TITLE_NOTE, ha='center', fontsize=9, color='#666666')
    fig.savefig('fig1_breakdown_stacked.png')
    fig.savefig('fig1_breakdown_stacked.pdf')
    plt.close(fig)
    print('  fig1 OK')


# ============================ 图2:增长曲线(核心) ============================
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (data, tag) in zip(axes, [(FP32, 'FP32'), (FP16, 'FP16')]):
        for s in SEGS:
            v = np.array(data[s])
            rel = v / v[0]
            ax.plot(PIX_REL, rel, 'o-', color=COLORS[s], lw=2.2, ms=7, label=s)
            ax.annotate(f'{rel[-1]:.2f}×', (PIX_REL[-1], rel[-1]),
                        textcoords='offset points', xytext=(8, 0),
                        color=COLORS[s], fontweight='bold', fontsize=10)
        # 参考线:理想线性(与像素数成正比)
        ax.plot(PIX_REL, PIX_REL, 'k:', lw=1.3, alpha=0.6, label='线性参考 (∝像素数)')
        ax.set_xlabel('相对像素数（以 608×960 为 1×）')
        ax.set_title(tag, fontweight='bold')
        ax.grid(alpha=0.3, ls=':')
        ax.set_axisbelow(True)
        ax.set_xticks(PIX_REL)
        ax.set_xticklabels([f'{p:.1f}×' for p in PIX_REL])
    axes[0].set_ylabel('相对耗时（各自归一化）')
    axes[1].legend(loc='upper left', fontsize=9, framealpha=0.95)
    fig.suptitle('图2  关键发现：Decoder 耗时与分辨率近似解耦，Backbone/Encoder 随像素数增长',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.text(0.5, -0.06, TITLE_NOTE, ha='center', fontsize=9, color='#666666')
    fig.savefig('fig2_scaling_curve.png')
    fig.savefig('fig2_scaling_curve.pdf')
    plt.close(fig)
    print('  fig2 OK')


# ============================ 图3:占比迁移 ============================
def fig3():
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(RES_LABELS))
    w = 0.36
    for k, (data, tag, hatch) in enumerate([(FP32, 'FP32', ''), (FP16, 'FP16', '//')]):
        tot = np.array([sum(data[s][i] for s in SEGS) for i in range(3)])
        dec = np.array(data['Decoder']) / tot * 100
        bars = ax.bar(x + (k - 0.5) * w, dec, w, label=f'Decoder 占比 ({tag})',
                      color='#C44E52', alpha=1.0 if k == 0 else 0.62,
                      hatch=hatch, edgecolor='white')
        for xi, v in zip(x + (k - 0.5) * w, dec):
            ax.text(xi, v + 0.9, f'{v:.1f}%', ha='center', fontsize=9.5, fontweight='bold')
    ax.axhline(25, color='#888888', ls='--', lw=1.2)
    ax.text(2.42, 25.9, '25% 参考线', fontsize=9, color='#666666', ha='right')
    ax.set_xticks(x); ax.set_xticklabels(RES_LABELS)
    ax.set_xlabel('输入分辨率'); ax.set_ylabel('Decoder 占总耗时比例 (%)')
    ax.set_ylim(0, 52)
    ax.grid(axis='y', alpha=0.3, ls=':'); ax.set_axisbelow(True)
    ax.legend(fontsize=9.5, framealpha=0.95)
    ax.set_title('图3  Decoder 占比随分辨率下降 —— 自适应解码的有效区间在中低分辨率',
                 fontsize=12, fontweight='bold', pad=12)
    fig.text(0.5, -0.04, TITLE_NOTE, ha='center', fontsize=9, color='#666666')
    fig.savefig('fig3_ratio_shift.png')
    fig.savefig('fig3_ratio_shift.pdf')
    plt.close(fig)
    print('  fig3 OK')


# ============================ 图4:节省潜力 ============================
def fig4():
    """理论节省潜力(基于 FP16 实测,按结构推算的上界)"""
    data, e2e = FP16, FP16_E2E
    tot = np.array([sum(data[s][i] for s in SEGS) for i in range(3)])
    bb = np.array(data['Backbone']); en = np.array(data['Encoder']); de = np.array(data['Decoder'])

    # 策略A:仅自适应解码(decoder 6层->1层,理论省 5/6 的 decoder)
    save_A = de * (5.0 / 6.0) / tot * 100
    # 策略B:查询传播(非关键帧跳过 backbone+encoder)
    save_B = (bb + en) / tot * 100
    # 策略C:两者结合
    save_C = (bb + en + de * (5.0 / 6.0)) / tot * 100

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(RES_LABELS)); w = 0.26
    sets = [
        (save_A, '策略A：仅自适应解码\n(decoder 6→1 层)', '#C44E52'),
        (save_B, '策略B：查询传播\n(非关键帧跳过 Backbone+Encoder)', '#4C72B0'),
        (save_C, '策略C：A + B 结合', '#55A868'),
    ]
    for k, (v, lab, c) in enumerate(sets):
        ax.bar(x + (k - 1) * w, v, w, label=lab, color=c, edgecolor='white')
        for xi, vi in zip(x + (k - 1) * w, v):
            ax.text(xi, vi + 1.2, f'{vi:.0f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(RES_LABELS)
    ax.set_xlabel('输入分辨率'); ax.set_ylabel('非关键帧理论节省比例 (%)')
    ax.set_ylim(0, 118)
    ax.grid(axis='y', alpha=0.3, ls=':'); ax.set_axisbelow(True)
    ax.legend(fontsize=8.6, loc='upper center', bbox_to_anchor=(0.5, -0.16),
              framealpha=0.95, ncol=3)
    ax.set_title('图4  不同策略在非关键帧上的理论节省上界（基于 FP16 实测推算）',
                 fontsize=12, fontweight='bold', pad=12)
    fig.text(0.5, -0.19,
             '注：为结构性上界，未计入传播/门控自身开销；实际收益需实验验证',
             ha='center', fontsize=9, color='#666666')
    fig.savefig('fig4_savings_potential.png')
    fig.savefig('fig4_savings_potential.pdf')
    plt.close(fig)
    print('  fig4 OK')


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print('生成图表中...')
    fig1(); fig2(); fig3(); fig4()
    print('\n全部完成，输出目录:', os.getcwd())
    for f in sorted(os.listdir('.')):
        if f.startswith('fig') and f.endswith('.png'):
            print(f'  {f}  ({os.path.getsize(f)/1024:.0f} KB)')
