# -*- coding: utf-8 -*-
"""
plot_grid.py —— 由 out_grid_2d.json 生成三张核心图 (300 DPI)
  figA_pareto_frontier.png : 精度-延迟散点 + 帕累托前沿 (色=层数, 蓝序数色带)
  figB_depth_vs_map.png    : 层数-mAP50 曲线 (每条线=一个查询数, 橙序数色带)
  figC_query_vs_map.png    : 查询数-mAP50 曲线 (每条线=一个层数, 蓝序数色带)
色带已过 validate_palette.js --ordinal; 形状作第二编码; 空心标记=0.2子集数据。
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, 'out_grid_2d.json'), encoding='utf-8'))
RES = DATA['results']
DEPTHS = DATA['meta']['depths']
QUERIES = DATA['meta']['queries']

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

# 验证过的序数色带
BLUE6 = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281', '#0a2547']
ORANGE5 = ['#eda266', '#eb6834', '#cd4f1a', '#a03a10', '#7a2c0c']
SHAPES6 = ['o', 's', '^', 'D', 'v', 'P']
SHAPES5 = ['o', 's', '^', 'D', 'v']

plt.rcParams.update({
    'font.family': 'Segoe UI',
    'font.size': 10,
    'text.color': INK,
    'axes.edgecolor': AXIS,
    'axes.labelcolor': INK2,
    'xtick.color': INK2,
    'ytick.color': INK2,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
})


def get(d, q, field):
    return RES.get(f'{d}_{q}', {}).get(field, np.nan)


def lat(d, q):
    """延迟: 优先 fused (部署口径), 否则 unfused"""
    r = RES.get(f'{d}_{q}', {})
    return r.get('p50_ms_f', r.get('p50_ms', np.nan))


FUSED = any('p50_ms_f' in r for r in RES.values())
LAT_NOTE = 'fused' if FUSED else 'unfused'


def best_map(d, q):
    """优先全量, 否则子集; 返回 (mAP50, 是否全量)"""
    r = RES.get(f'{d}_{q}', {})
    if 'map50_full' in r:
        return r['map50_full'], True
    return r.get('map50_sub', np.nan), False


def chrome(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(1)
    ax.grid(axis='y', color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def legend_style(ax, title, loc='best'):
    lg = ax.legend(title=title, loc=loc, frameon=False, fontsize=9,
                   title_fontsize=9, labelspacing=0.35, handletextpad=0.6,
                   borderaxespad=0.2)
    lg.get_title().set_color(INK2)
    for t in lg.get_texts():
        t.set_color(INK2)
    return lg


# ---------------- 图 A: 精度-延迟散点 + 帕累托前沿 ----------------
def fig_a():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    pts = []  # (lat, map, d, q, full)
    for i, d in enumerate(DEPTHS):
        lat_ = [lat(d, q) for q in QUERIES]
        mp = [best_map(d, q)[0] for q in QUERIES]
        fl = [best_map(d, q)[1] for q in QUERIES]
        # 细连线帮助分组 (latency 随查询数单调增)
        ax.plot(lat_, mp, color=BLUE6[i], linewidth=1.5, alpha=0.55, zorder=2)
        for q, la, m, f in zip(QUERIES, lat_, mp, fl):
            pts.append((la, m, d, q, f))
            ax.scatter(la, m, s=64, marker=SHAPES6[i], zorder=3,
                       facecolors=(BLUE6[i] if f else 'none'),
                       edgecolors=(SURFACE if f else BLUE6[i]),
                       linewidths=(1.4 if f else 1.6))
        # 图例代理
        ax.scatter([], [], s=64, marker=SHAPES6[i], color=BLUE6[i],
                   edgecolors=SURFACE, linewidths=1.4, label=f'{d}')

    # 帕累托前沿: latency 越小越好, mAP 越大越好
    pts.sort(key=lambda p: (p[0], -p[1]))
    frontier, best = [], -np.inf
    for p in pts:
        if p[1] > best:
            frontier.append(p)
            best = p[1]
    fx = [p[0] for p in frontier]
    fy = [p[1] for p in frontier]
    ax.plot(fx, fy, color=INK, linewidth=2, zorder=4,
            solid_joinstyle='round', solid_capstyle='round',
            label='Pareto frontier')
    ax.scatter(fx, fy, s=30, color=INK, zorder=5)
    # 标签: 每层只标前沿上 mAP 最高的点, 其余黑点不标 (全部配置见报告表格)
    best_of_depth = {}
    for p in frontier:
        if p[2] not in best_of_depth or p[1] > best_of_depth[p[2]][1]:
            best_of_depth[p[2]] = p
    for i, (la, m, d, q, _) in enumerate(best_of_depth.values()):
        off = (7, 7) if i % 2 == 0 else (8, -14)
        ax.annotate(f'{d}/{q}', (la, m), textcoords='offset points',
                    xytext=off, fontsize=8.5, color=INK, zorder=6)

    # 15 ms 预算线
    ax.axvline(15, color=MUTED, linewidth=1, zorder=1)

    ax.set_xlabel(f'Median latency (ms, batch 1, {LAT_NOTE} pure forward)')
    ax.set_ylabel('mAP@0.5')
    ax.set_title('RT-DETR R18: accuracy-latency trade-off, 30 configs (Argoverse-HD val)',
                 fontsize=11, color=INK, pad=10)
    lg = ax.legend(title='Decoder depth', ncol=2, loc='upper left',
                   bbox_to_anchor=(0.02, 0.98),
                   frameon=False, fontsize=9, title_fontsize=9,
                   labelspacing=0.35, handletextpad=0.4, columnspacing=0.8)
    lg.get_title().set_color(INK2)
    for t in lg.get_texts():
        t.set_color(INK2)
    chrome(ax)
    y1 = ax.get_ylim()[1]
    ax.text(14.85, y1, '15 ms budget ', fontsize=8.5, color=MUTED,
            va='top', ha='right')
    fig.text(0.01, 0.005, 'Open markers: 0.2-subset eval; filled: full val. Label = depth/queries.',
             fontsize=8, color=MUTED)
    fig.savefig(os.path.join(HERE, 'figA_pareto_frontier.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    return frontier


# ---------------- 图 B: 层数-mAP50, 每线一个查询数 ----------------
def fig_b():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for j, q in enumerate(QUERIES):
        ys = [best_map(d, q)[0] for d in DEPTHS]
        fl = [best_map(d, q)[1] for d in DEPTHS]
        ax.plot(DEPTHS, ys, color=ORANGE5[j], linewidth=2,
                solid_joinstyle='round', solid_capstyle='round', zorder=2)
        for d, y, f in zip(DEPTHS, ys, fl):
            ax.scatter(d, y, s=64, marker=SHAPES5[j], zorder=3,
                       facecolors=(ORANGE5[j] if f else 'none'),
                       edgecolors=(SURFACE if f else ORANGE5[j]),
                       linewidths=1.4)
        ax.scatter([], [], s=64, marker=SHAPES5[j], color=ORANGE5[j],
                   edgecolors=SURFACE, linewidths=1.4, label=f'{q}')
    ax.set_xticks(DEPTHS)
    ax.set_xlabel('Decoder depth (eval layers)')
    ax.set_ylabel('mAP@0.5')
    ax.set_title('Depth vs accuracy, one line per query count', fontsize=11,
                 color=INK, pad=10)
    legend_style(ax, 'Queries')
    chrome(ax)
    fig.text(0.01, 0.005, 'Open markers: 0.2-subset eval; filled: full val.',
             fontsize=8, color=MUTED)
    fig.savefig(os.path.join(HERE, 'figB_depth_vs_map.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------- 图 C: 查询数-mAP50, 每线一个层数 ----------------
def fig_c():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, d in enumerate(DEPTHS):
        ys = [best_map(d, q)[0] for q in QUERIES]
        fl = [best_map(d, q)[1] for q in QUERIES]
        ax.plot(QUERIES, ys, color=BLUE6[i], linewidth=2,
                solid_joinstyle='round', solid_capstyle='round', zorder=2)
        for q, y, f in zip(QUERIES, ys, fl):
            ax.scatter(q, y, s=64, marker=SHAPES6[i], zorder=3,
                       facecolors=(BLUE6[i] if f else 'none'),
                       edgecolors=(SURFACE if f else BLUE6[i]),
                       linewidths=1.4)
        ax.scatter([], [], s=64, marker=SHAPES6[i], color=BLUE6[i],
                   edgecolors=SURFACE, linewidths=1.4, label=f'{d}')
    ax.set_xticks(QUERIES)
    ax.set_xlabel('Number of decoder queries')
    ax.set_ylabel('mAP@0.5')
    ax.set_title('Query count vs accuracy, one line per decoder depth',
                 fontsize=11, color=INK, pad=10)
    lg = ax.legend(title='Decoder depth', loc='center left',
                   bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9,
                   title_fontsize=9, labelspacing=0.35, handletextpad=0.6)
    lg.get_title().set_color(INK2)
    for t in lg.get_texts():
        t.set_color(INK2)
    chrome(ax)
    fig.text(0.01, 0.005, 'Open markers: 0.2-subset eval; filled: full val.',
             fontsize=8, color=MUTED)
    fig.savefig(os.path.join(HERE, 'figC_query_vs_map.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fr = fig_a()
    fig_b()
    fig_c()
    print('Pareto frontier (lat, mAP50, d/q):')
    for la, m, d, q, f in fr:
        print(f'  {la:6.2f} ms  {m:.4f}  {d}/{q}  {"full" if f else "sub"}')
    print('figA/B/C 已生成')
