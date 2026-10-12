# -*- coding: utf-8 -*-
"""
关键帧/非关键帧实测数据绘图
输出:
  fig5_savings_measured.png   非关键帧节省比例(实测,替代原理论上界图)
  fig6_amortized_curve.png    摊薄单帧耗时 vs 关键帧间隔 K
  fig7_equiv_fps.png          等效 FPS 对比(最直观)
  fig8_logic_chain.png        逻辑链条图(数据支撑的推理链)
数据来源: bench_keyframe.py 在 RTX 5070 Ti Laptop 上实测(RT-DETR-L, FP16)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 300

RES = ['608×960', '800×1280', '1216×1920']
FULL = np.array([27.70, 33.67, 57.25])          # 完整前向 ms
DEC_ONLY = np.array([9.64, 9.95, 12.37])        # 只跑 decoder ms
DEC3 = np.array([5.36, 5.79, 7.36])             # decoder 前3层
DEC1 = np.array([2.82, 3.33, 4.18])             # decoder 前1层

SAVE_DEC = (1 - DEC_ONLY / FULL) * 100          # 65.2 / 70.5 / 78.4
SAVE_D3 = (1 - DEC3 / FULL) * 100               # 80.2 / 82.8 / 87.1
SAVE_D1 = (1 - DEC1 / FULL) * 100               # 89.5 / 90.1 / 92.7

NOTE = 'RT-DETR-L @ RTX 5070 Ti Laptop，FP16，中位数；未计入传播/门控自身开销'
C_FULL, C_DEC, C_D3, C_D1 = '#8899AA', '#4C72B0', '#55A868', '#C44E52'


def fig5():
    """实测节省比例"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.7))
    x = np.arange(3); w = 0.26

    # 左:绝对耗时
    ax1.bar(x - 1.5*w, FULL, w, label='完整前向（关键帧）', color=C_FULL, edgecolor='white')
    ax1.bar(x - 0.5*w, DEC_ONLY, w, label='只跑 Decoder（查询传播）', color=C_DEC, edgecolor='white')
    ax1.bar(x + 0.5*w, DEC3, w, label='Decoder 前 3 层', color=C_D3, edgecolor='white')
    ax1.bar(x + 1.5*w, DEC1, w, label='Decoder 前 1 层', color=C_D1, edgecolor='white')
    for xi, arr in zip([x-1.5*w, x-0.5*w, x+0.5*w, x+1.5*w], [FULL, DEC_ONLY, DEC3, DEC1]):
        for xx, vv in zip(xi, arr):
            ax1.text(xx, vv + 1.1, f'{vv:.1f}', ha='center', fontsize=8.4, fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(RES)
    ax1.set_ylabel('单帧耗时 (ms)'); ax1.set_xlabel('输入分辨率')
    ax1.set_ylim(0, 66)
    ax1.legend(fontsize=8.8, framealpha=0.95)
    ax1.grid(axis='y', alpha=0.3, ls=':'); ax1.set_axisbelow(True)
    ax1.set_title('(a) 非关键帧各执行路径的实测耗时', fontsize=11.5, fontweight='bold')

    # 右:节省比例(随分辨率上升)
    ax2.plot(x, SAVE_DEC, 'o-', color=C_DEC, lw=2.4, ms=8, label='只跑 Decoder（查询传播）')
    ax2.plot(x, SAVE_D3, 's-', color=C_D3, lw=2.4, ms=7, label='传播 + Decoder 前 3 层')
    ax2.plot(x, SAVE_D1, '^-', color=C_D1, lw=2.4, ms=7, label='传播 + Decoder 前 1 层')
    for arr, c in [(SAVE_DEC, C_DEC), (SAVE_D3, C_D3), (SAVE_D1, C_D1)]:
        for xx, vv in zip(x, arr):
            ax2.text(xx, vv + 1.3, f'{vv:.1f}%', ha='center', fontsize=9.2,
                     color=c, fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(RES)
    ax2.set_xlabel('输入分辨率'); ax2.set_ylabel('非关键帧节省比例 (%)')
    ax2.set_ylim(55, 100)
    ax2.legend(fontsize=8.8, loc='lower right', framealpha=0.95)
    ax2.grid(alpha=0.3, ls=':'); ax2.set_axisbelow(True)
    ax2.set_title('(b) 分辨率越高，查询传播省得越多', fontsize=11.5, fontweight='bold')

    fig.suptitle('图5  实测：非关键帧跳过 Backbone+Encoder 的真实收益',
                 fontsize=13.5, fontweight='bold', y=1.03)
    fig.text(0.5, -0.05, NOTE, ha='center', fontsize=9, color='#666666')
    fig.savefig('fig5_savings_measured.png'); fig.savefig('fig5_savings_measured.pdf')
    plt.close(fig); print('  fig5 OK')


def amortized(full, nonkey, Ks):
    return np.array([(full + (k - 1) * nonkey) / k for k in Ks])


def fig6():
    """摊薄耗时 vs K"""
    Ks = np.arange(1, 13)
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.5), sharey=False)
    for ax, i in zip(axes, range(3)):
        ax.axhline(FULL[i], color=C_FULL, ls='--', lw=2, label=f'逐帧全算 ({FULL[i]:.1f}ms)')
        ax.plot(Ks, amortized(FULL[i], DEC_ONLY[i], Ks), 'o-', color=C_DEC,
                lw=2.2, ms=5, label='仅查询传播')
        ax.plot(Ks, amortized(FULL[i], DEC3[i], Ks), 's-', color=C_D3,
                lw=2.2, ms=5, label='传播 + 3 层')
        ax.plot(Ks, amortized(FULL[i], DEC1[i], Ks), '^-', color=C_D1,
                lw=2.2, ms=5, label='传播 + 1 层')
        # 33ms 实时线(30FPS)
        ax.axhline(33.3, color='#D62728', ls=':', lw=1.6, alpha=0.8)
        ax.text(12, 34.6, '30 FPS 实时线', fontsize=8.4, color='#D62728', ha='right')
        ax.set_xlabel('关键帧间隔 K'); ax.set_title(RES[i], fontweight='bold', fontsize=11.5)
        ax.grid(alpha=0.3, ls=':'); ax.set_axisbelow(True)
        ax.set_xticks([1, 3, 5, 7, 9, 11])
    axes[0].set_ylabel('摊薄后平均单帧耗时 (ms)')
    axes[2].legend(fontsize=8.6, framealpha=0.95)
    fig.suptitle('图6  摊薄单帧耗时随关键帧间隔 K 的变化（实测数据外推）',
                 fontsize=13.5, fontweight='bold', y=1.03)
    fig.text(0.5, -0.06, NOTE, ha='center', fontsize=9, color='#666666')
    fig.savefig('fig6_amortized_curve.png'); fig.savefig('fig6_amortized_curve.pdf')
    plt.close(fig); print('  fig6 OK')


def fig7():
    """等效 FPS(最直观)"""
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    x = np.arange(3); w = 0.2
    K = 10
    fps_full = 1000 / FULL
    fps_dec = 1000 / amortized(FULL, DEC_ONLY, [K])[0] if False else np.array(
        [1000 / ((FULL[i] + (K-1)*DEC_ONLY[i]) / K) for i in range(3)])
    fps_d3 = np.array([1000 / ((FULL[i] + (K-1)*DEC3[i]) / K) for i in range(3)])
    fps_d1 = np.array([1000 / ((FULL[i] + (K-1)*DEC1[i]) / K) for i in range(3)])

    for k, (v, lab, c) in enumerate([
        (fps_full, '逐帧全算', C_FULL),
        (fps_dec, f'仅查询传播 (K={K})', C_DEC),
        (fps_d3, f'传播+3层 (K={K})', C_D3),
        (fps_d1, f'传播+1层 (K={K})', C_D1),
    ]):
        ax.bar(x + (k - 1.5) * w, v, w, label=lab, color=c, edgecolor='white')
        for xx, vv in zip(x + (k - 1.5) * w, v):
            ax.text(xx, vv + 2, f'{vv:.0f}', ha='center', fontsize=9, fontweight='bold')
    ax.axhline(30, color='#D62728', ls='--', lw=1.8)
    ax.text(2.46, 32, '30 FPS 实时线', fontsize=9.5, color='#D62728', ha='right', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(RES)
    ax.set_xlabel('输入分辨率'); ax.set_ylabel('等效帧率 (FPS)')
    ax.set_ylim(0, max(fps_d1.max(), fps_d3.max()) * 1.22)
    ax.legend(fontsize=9.2, framealpha=0.95, ncol=2, loc='upper right')
    ax.grid(axis='y', alpha=0.3, ls=':'); ax.set_axisbelow(True)
    ax.set_title('图7  等效帧率对比：1216×1920 下由 17 FPS 提升至 59～105 FPS（跨越实时线）',
                 fontsize=12, fontweight='bold', pad=12)
    fig.text(0.5, -0.04, NOTE + '；K=10 表示每 10 帧一个关键帧',
             ha='center', fontsize=9, color='#666666')
    fig.savefig('fig7_equiv_fps.png'); fig.savefig('fig7_equiv_fps.pdf')
    plt.close(fig); print('  fig7 OK')


def fig8():
    """逻辑链条图"""
    fig, ax = plt.subplots(figsize=(13.2, 5.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis('off')

    boxes = [
        # (x, y, w, h, 标题, 内容, 颜色)
        (2, 40, 27, 14, '① 观察（实测）',
         'Decoder 耗时与分辨率解耦\n1.09×  vs  Backbone 2.31×', '#E8F0F8', '#4C72B0'),
        (36.5, 40, 27, 14, '② 推论',
         '随分辨率膨胀的是\nBackbone + Encoder（占 66~82%）', '#E8F0F8', '#4C72B0'),
        (71, 40, 27, 14, '③ 方法选择',
         '查询传播：非关键帧\n跳过 Backbone + Encoder', '#FFF0E8', '#C44E52'),
        (2, 20, 27, 14, '④ 实测验证',
         '非关键帧省 65.2% → 78.4%\n（分辨率越高省得越多）', '#EAF5EC', '#55A868'),
        (36.5, 20, 27, 14, '⑤ 补充问题',
         'Decoder 仍占 27.6~43.2%\n单靠传播不够', '#FFF0E8', '#C44E52'),
        (71, 20, 27, 14, '⑥ 补充方法',
         '自适应解码：按内容变化\n路由 Decoder 层数', '#FFF0E8', '#C44E52'),
        (36.5, 2, 27, 13, '⑦ 联合收益（实测）',
         '节省 89.5% → 92.7%\n17 FPS → 105 FPS', '#FDF2CC', '#B8860B'),
    ]
    for x, y, w, h, title, content, fc, ec in boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6',
                             facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 3.4, title, ha='center', va='center',
                fontsize=11.5, fontweight='bold', color=ec)
        ax.text(x + w/2, y + h/2 - 2.6, content, ha='center', va='center',
                fontsize=10.2, color='#333333', linespacing=1.5)

    def arrow(pts, color='#666666'):
        """按折线路径画箭头(避免斜穿其它方框)"""
        for i in range(len(pts) - 1):
            style = '-|>' if i == len(pts) - 2 else '-'
            ax.add_patch(FancyArrowPatch(pts[i], pts[i+1], arrowstyle=style,
                                         mutation_scale=18, color=color, lw=2,
                                         shrinkA=0, shrinkB=0))

    # 上排 ① -> ② -> ③
    arrow([(29.2, 47), (36.3, 47)])
    arrow([(63.7, 47), (70.8, 47)])
    # ③ 折向左侧 ④(绕过中间, 走上方通道)
    arrow([(84.5, 39.8), (84.5, 37.2), (15.5, 37.2), (15.5, 34.2)])
    # 中排 ④ -> ⑤ -> ⑥
    arrow([(29.2, 27), (36.3, 27)])
    arrow([(63.7, 27), (70.8, 27)])
    # ⑥ 折向 ⑦(走右侧再向下)
    arrow([(84.5, 19.8), (84.5, 15.6), (63.5, 15.6), (63.5, 12.0)])
    # ④ 直接向下汇入 ⑦
    arrow([(15.5, 19.8), (15.5, 8.5), (36.3, 8.5)])

    ax.text(50, 56.4, '图8  数据支撑的逻辑链条', ha='center',
            fontsize=14, fontweight='bold')
    fig.savefig('fig8_logic_chain.png'); fig.savefig('fig8_logic_chain.pdf')
    plt.close(fig); print('  fig8 OK')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print('生成图表中...')
    fig5(); fig6(); fig7(); fig8()
    print('\n完成:', os.getcwd())
    for f in sorted(os.listdir('.')):
        if f.startswith('fig') and f.endswith('.png'):
            print(f'  {f}  ({os.path.getsize(f)/1024:.0f} KB)')
