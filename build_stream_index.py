# -*- coding: utf-8 -*-
"""
build_stream_index.py —— 从平铺的 Argoverse-HD 图片重建时序流索引
==================================================================
背景:
    Argoverse-HD 的图片下载后是平铺在 images/ 下的, 文件名形如
        ring_front_center_315966774109236128.jpg
    其中 315966774109236128 是 19 位纳秒时间戳。但下载/解压顺序与
    时间顺序无关, 直接用 os.listdir() 或按文件名排序得到的"帧序列"
    是乱序的 —— 此时任何"上一帧/当前帧"的假设(运动补偿、时序传播、
    motion IoU)全部失效。

本脚本做三件事:
    1. 解析文件名中的纳秒时间戳, 按时间排序
    2. 在相邻间隔 > --gap-s 处切分为独立序列(scene)
    3. 输出 stream_index.json, 供流式评测与实验脚本按真实时序取帧

输出 stream_index.json 结构:
    {
      "camera": "ring_front_center",
      "gap_s": 1.0,
      "n_frames": 15062,
      "n_scenes": 21,
      "scenes": [
        {"scene_id": 0, "start_ns":..., "end_ns":..., "n_frames": 812,
         "duration_s": 27.1, "fps_median": 30.0,
         "frames": [{"idx":0, "file":"...jpg", "t_ns":..., "dt_ms":null}, ...]},
        ...
      ]
    }

    每条 frame 的 t_ns 是绝对纳秒时间戳; idx 是该 scene 内的帧序号(从 0 起);
    dt_ms 是与前一顿的间隔毫秒(首帧为 null)。

用法:
    # 默认扫描本地 val 图片目录
    python build_stream_index.py

    # 指定目录与输出
    python build_stream_index.py --images 数据集/argoverse-hd/images \
                                --out 数据集/argoverse-hd/stream_index.json

    # 只做诊断不写文件
    python build_stream_index.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ring_front_center_315966774109236128.jpg  ->  相机名 + 19位纳秒时间戳
NAME_RE = re.compile(r'^(?P<cam>.+)_(?P<ts>\d{15,20})$')


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- 扫描

def load_split_files(ndjson, split):
    """从 ndjson 读出某个 split 的文件名集合; split 为 None 时返回 None(不过滤)"""
    if not split:
        return None
    keep = set()
    with open(ndjson, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get('type') == 'image' and o.get('split') == split:
                keep.add(o['file'])
    return keep


def scan_frames(img_dir, keep=None):
    """扫描目录, 返回 (frames, bad)

    frames: [{'file':..., 'cam':..., 't_ns':...}, ...] 已按时间升序
    bad   : 文件名无法解析的列表
    keep  : 若给出, 只保留文件名在此集合中的帧(用于按 split 拆时序)
    """
    frames, bad = [], []
    for fn in os.listdir(img_dir):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        if keep is not None and fn not in keep:
            continue
        m = NAME_RE.match(stem)
        if not m:
            bad.append(fn)
            continue
        frames.append({'file': fn, 'cam': m.group('cam'), 't_ns': int(m.group('ts'))})
    frames.sort(key=lambda x: x['t_ns'])
    return frames, bad


def check_cameras(frames):
    """返回 (主相机, 全部相机计数)。多相机时只保留主相机, 其余打警告。"""
    c = Counter(f['cam'] for f in frames)
    main = c.most_common(1)[0][0]
    return main, c


# ---------------------------------------------------------------- 切分

def split_scenes(frames, gap_ns):
    """按相邻时间戳间隔切分, 返回 scene 列表(每个是 frames 的子列表)"""
    scenes, cur = [], []
    for i, f in enumerate(frames):
        if i > 0 and f['t_ns'] - frames[i - 1]['t_ns'] > gap_ns:
            scenes.append(cur)
            cur = []
        cur.append(f)
    if cur:
        scenes.append(cur)
    return scenes


def build_index(frames, gap_s):
    """把 frames 组装成 stream_index 结构"""
    gap_ns = int(gap_s * 1e9)
    scenes_raw = split_scenes(frames, gap_ns)

    scenes = []
    for sid, raw in enumerate(scenes_raw):
        # 丢弃长度 1 的碎场景: 单帧无法构成时序, 且会让 scene 统计失真
        fr = []
        for i, f in enumerate(raw):
            dt = None if i == 0 else round((f['t_ns'] - raw[i - 1]['t_ns']) / 1e6, 3)
            fr.append({'idx': i, 'file': f['file'], 't_ns': f['t_ns'], 'dt_ms': dt})
        dts = [x['dt_ms'] for x in fr[1:] if x['dt_ms'] is not None]
        med = sorted(dts)[len(dts) // 2] if dts else None
        scenes.append({
            'scene_id': sid,
            'start_ns': raw[0]['t_ns'],
            'end_ns': raw[-1]['t_ns'],
            'n_frames': len(fr),
            'duration_s': round((raw[-1]['t_ns'] - raw[0]['t_ns']) / 1e9, 3),
            'fps_median': round(1000.0 / med, 2) if med else None,
            'frames': fr,
        })
    return scenes


# ---------------------------------------------------------------- 报告

def report(frames, scenes, gap_s, bad, cams):
    n = len(frames)
    log('')
    log('=' * 62)
    log('Argoverse-HD 时序流索引 —— 诊断报告')
    log('=' * 62)
    log(f'可解析帧数        : {n}')
    if bad:
        log(f'无法解析文件名    : {len(bad)}  (示例 {bad[:3]})')
    log(f'相机              : {dict(cams)}')

    if n == 0:
        return

    span = (frames[-1]['t_ns'] - frames[0]['t_ns']) / 1e9
    log(f'时间跨度          : {span:.1f} s  ({span/3600:.2f} h)')
    log(f'整体平均帧率      : {n/span:.2f} FPS  (含场景间隙, 会偏低)')

    # 间隔直方图: 确认切分阈值是否落在安全的两不管地带
    gaps_ms = [(b['t_ns'] - a['t_ns']) / 1e6 for a, b in zip(frames, frames[1:])]
    intra = [g for g in gaps_ms if g <= gap_s * 1000]
    inter = [g for g in gaps_ms if g > gap_s * 1000]
    log('')
    log(f'--- 帧间隔 (切分阈值 {gap_s:.1f} s) ---')
    if intra:
        si = sorted(intra)
        log(f'  场景内间隔  n={len(intra):<6}  '
            f'min={si[0]:.1f}  p50={si[len(si)//2]:.1f}  max={si[-1]:.1f} ms')
    log(f'  场景间断裂  n={len(inter):<6}  '
        + (f'min={min(inter)/1000:.1f} s  max={max(inter)/1000:.1f} s' if inter else '无'))

    # 阈值安全边界: 场景内最大间隔 vs 场景间最小间隔, 相差越大越安全
    if intra and inter:
        log(f'  安全余量    : 场景内最大 {max(intra)/1000:.3f} s  vs  '
            f'场景间最小 {min(inter)/1000:.1f} s  ->  '
            f'{min(inter)/max(intra):.0f} 倍')

    # 帧率档位: Argoverse-HD 混合了 30/50/70 FPS 的录制段, 影响流式评测的时间口径
    log('')
    log('--- 帧间隔档位分布 (决定流式评测的时间口径) ---')
    buckets = Counter()
    for g in intra:
        if g < 16:
            buckets['~14ms (70FPS)'] += 1
        elif g < 25:
            buckets['~19ms (50FPS)'] += 1
        elif g < 45:
            buckets['~33ms (30FPS)'] += 1
        else:
            buckets[f'>{gap_s*1000:.0f}ms 其它'] += 1
    for k, v in buckets.most_common():
        log(f'  {k:<16} {v:>6}  ({v/max(1,len(intra))*100:5.1f}%)')

    # 场景统计
    log('')
    log(f'--- 场景切分结果: {len(scenes)} 段 ---')
    if scenes:
        lens = sorted(s['n_frames'] for s in scenes)
        log(f'  每段帧数  min={lens[0]}  p50={lens[len(lens)//2]}  max={lens[-1]}  '
            f'合计={sum(lens)}')
        solo = [s for s in scenes if s['n_frames'] < 10]
        log(f'  短序列(<10帧): {len(solo)} 段'
            + (f'  -> scene_id {[s["scene_id"] for s in solo][:10]}' if solo else ''))
        log('')
        log('  scene_id  帧数    时长(s)   中位FPS   起始时间戳')
        show = scenes if len(scenes) <= 30 else scenes[:15] + scenes[-5:]
        for s in show:
            if s is None:
                continue
            log(f'  {s["scene_id"]:>8}  {s["n_frames"]:>5}   {s["duration_s"]:>8.1f}   '
                f'{str(s["fps_median"]):>7}   {s["start_ns"]}')
        if len(scenes) > 30:
            log(f'  ... 中间省略 {len(scenes)-20} 段 ...')

    # 对下游实验最关键的一个数: 可用于时序传播的相邻帧对数量
    pairs = sum(max(0, s['n_frames'] - 1) for s in scenes)
    log('')
    log(f'>>> 可用相邻帧对: {pairs}  (乱序排序下这一保证不存在)')


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default=r'D:\lunwen\自研\数据集\argoverse-hd\images',
                    help='图片目录')
    ap.add_argument('--out', default=r'D:\lunwen\自研\数据集\argoverse-hd\stream_index.json',
                    help='输出 json 路径')
    ap.add_argument('--gap-s', type=float, default=1.0,
                    help='相邻帧间隔超过该秒数则切分为新场景 (默认 1.0)')
    ap.add_argument('--split', default=None, choices=['train', 'val'],
                    help='只对该 split 建索引(train 与 val 的图片在同一目录, 必须分开)')
    ap.add_argument('--ndjson', default=r'D:\lunwen\自研\数据集\argoverse-hd.ndjson',
                    help='ndjson 路径, 配合 --split 用于筛出该 split 的文件名')
    ap.add_argument('--min-frames', type=int, default=2,
                    help='帧数少于该值的场景被丢弃 (默认 2, 即丢掉孤立帧)')
    ap.add_argument('--dry-run', action='store_true', help='只诊断, 不写文件')
    args = ap.parse_args()

    if not os.path.isdir(args.images):
        log(f'[错误] 图片目录不存在: {args.images}')
        sys.exit(1)

    keep = load_split_files(args.ndjson, args.split)
    if keep is not None:
        log(f'[1/4] 扫描 {args.images}  (仅 split={args.split}, ndjson 声明 {len(keep)} 张)')
    else:
        log(f'[1/4] 扫描 {args.images}  (全部)')
    frames, bad = scan_frames(args.images, keep)
    log(f'      解析到 {len(frames)} 帧, 失败 {len(bad)}')
    if keep is not None and len(frames) != len(keep):
        log(f'      [警告] ndjson 有 {len(keep)} 张, 本地只找到 {len(frames)} 张, '
            f'差 {len(keep)-len(frames)}(可能未下载完)')
    if not frames:
        log('[错误] 没有解析到任何帧, 检查文件名格式')
        sys.exit(1)

    main_cam, cams = check_cameras(frames)
    if len(cams) > 1:
        log(f'      [警告] 发现多路相机 {dict(cams)}, 只保留 {main_cam}')
        frames = [f for f in frames if f['cam'] == main_cam]

    log(f'[2/4] 按时间戳排序并切分 (gap>{args.gap_s}s)')
    scenes = build_index(frames, args.gap_s)

    n_before = len(scenes)
    scenes = [s for s in scenes if s['n_frames'] >= args.min_frames]
    if n_before != len(scenes):
        log(f'      丢弃 {n_before-len(scenes)} 个短于 {args.min_frames} 帧的场景')
        for i, s in enumerate(scenes):   # scene_id 重排, 保证连续
            s['scene_id'] = i

    kept = sum(s['n_frames'] for s in scenes)
    if kept != len(frames):
        log(f'      [注意] 保留下来的帧 {kept} != 扫描帧 {len(frames)}, '
            f'差额为被丢弃的孤立帧')

    log('[3/4] 生成诊断报告')
    report(frames, scenes, args.gap_s, bad, cams)

    if args.dry_run:
        log('')
        log('[4/4] --dry-run, 不写文件')
        return

    out = {
        'source': 'Argoverse-HD (ultralytics ndjson 导出)',
        'split': args.split or 'all',
        'camera': main_cam,
        'gap_s': args.gap_s,
        'n_frames': kept,
        'n_scenes': len(scenes),
        'scenes': scenes,
    }
    log('')
    log(f'[4/4] 写入 {args.out}')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    log(f'      完成: {kept} 帧 / {len(scenes)} 场景, '
        f'{os.path.getsize(args.out)/1e6:.1f} MB')

    # 抽查: 打印第一个长场景的头 3 帧, 人工确认时间戳是递增的
    big = max(scenes, key=lambda s: s['n_frames'])
    log('')
    log(f'抽查最长的场景 scene_id={big["scene_id"]} 前 3 帧:')
    for fr in big['frames'][:3]:
        log(f'  idx={fr["idx"]}  dt={fr["dt_ms"]}ms  {fr["file"]}')


if __name__ == '__main__':
    main()
