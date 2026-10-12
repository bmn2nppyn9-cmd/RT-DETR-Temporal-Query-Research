# -*- coding: utf-8 -*-
"""
eval_grid_3d.py —— RT-DETR 三维消融网格: depth × queries × resolution
=====================================================================
在二维消融 (ablation_2d, 6层 × 5查询) 基础上加第三维: 输入分辨率。
目标: 15ms 延迟预算下的静态最优 (depth, queries, resolution)。

网格: depths {1,3,6} × queries {150,300,375} × resolutions {480,512,640} = 27 配置。
同一权重, 纯推理期配置调整 (eval_idx / num_queries / imgsz), 不重训。

阶段 (位置参数):
  timing  : 27 配置交错测延迟 —— 每配置 50 预热 + 500 计时
            (50 轮 × 每轮每配置 10 次), 另按分辨率实测 GFLOPs       (~10 分钟)
  scan    : 27 配置 × fraction=0.2 (3012 帧) 子集快扫精度           (~80 分钟)
  full    : 关键配置全量 val。r=640 且二维已有全量结果的配置直接导入
            (同权重同协议, 不重复跑); 其余逐个 val                  (~30-60 分钟)
  merge   : 合并三个 JSON -> ablation_3d_results.csv + 预算前沿表
  all     : timing -> scan -> full -> merge

配置选择 (full 阶段):
  命令行:  python eval_grid_3d.py full --full-configs "3,375,512" "1,300,480"
  无参数时自动选 = 用户锚点单 ∪ 每个 (depth,res) 下扫描最优 queries
                 且 fused 中位延迟 <= 预算+1ms 的配置 (上限 9 个)。

输出 (均在脚本所在目录):
  timing_3d.json / scan_3d_02.json / full_3d.json / ablation_3d_results.csv
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # Windows conda OMP 冲突规避

import argparse
import json
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

WORKDIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(WORKDIR)          # D:\lunwen\自研
os.chdir(ROOT)                           # 相对路径锚到项目根

# ============ 配置 ============
MODEL_PATH = 'runs/detect/runs_rtdetr/r18_640_sub20_b4/weights/best.pt'
DATA_YAML = 'cfg/argoverse-hd-local.yaml'
GRID_2D_JSON = os.path.join(ROOT, 'ablation_2d', 'out_grid_2d.json')  # 二维全量复用源
SEED = 20260917
BUDGET_MS = 15.0

DEPTHS = [1, 3, 6]
QUERIES = [150, 300, 375]
RESOLUTIONS = [480, 512, 640]

TIMING_JSON = os.path.join(WORKDIR, 'timing_3d.json')
SCAN_JSON = os.path.join(WORKDIR, 'scan_3d_02.json')
FULL_JSON = os.path.join(WORKDIR, 'full_3d.json')
CSV_OUT = os.path.join(WORKDIR, 'ablation_3d_results.csv')

# 用户锚点 (r=640 的三个在二维已有全量, 会被导入而非重跑)
DEFAULT_FULL = [(1, 375, 640), (3, 375, 640), (6, 300, 640),
                (3, 375, 512), (3, 300, 512), (1, 300, 480)]


def log(m):
    print(m, flush=True)


def key(d, q, r):
    return f'd{d}_q{q}_r{r}'


def load_json(path):
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_json(data, path):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def find_decoder(model):
    """定位 RTDETRDecoder (比 model.model.model[-1] 硬路径稳)"""
    for mod in model.modules():
        if type(mod).__name__ == 'RTDETRDecoder' and hasattr(mod, 'num_queries'):
            return mod
    return None


def set_cfg(dec, d, q):
    dec.decoder.eval_idx = d - 1     # transformer.py:792 推理时提前退出
    dec.num_queries = q              # head.py:1709/1606 运行时读取


# ============ 阶段 1: 延迟测量 (27 配置交错) ============
def stage_timing():
    import torch
    from ultralytics import RTDETR
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    m = RTDETR(MODEL_PATH)
    dec = find_decoder(m.model)
    net = m.model.cuda().eval()
    net.fuse()                       # 部署口径, 与二维 fused 列一致
    log(f'[timing] fused, params={get_num_params(net)/1e6:.2f}M')

    # 每个分辨率预建输入 (避免计时间隙的分配噪声)
    xs = {r: torch.rand(1, 3, r, r, device='cuda') for r in RESOLUTIONS}
    cfgs = [(d, q, r) for d in DEPTHS for q in QUERIES for r in RESOLUTIONS]
    times = {key(d, q, r): [] for d, q, r in cfgs}

    log('[timing] 预热 每配置50次 ...')
    with torch.no_grad():
        for d, q, r in cfgs:
            set_cfg(dec, d, q)
            for _ in range(50):
                net(xs[r])
    torch.cuda.synchronize()

    log('[timing] 计时 50轮 x 27配置 x 10次 = 每配置500次 (交错) ...')
    t0 = time.time()
    with torch.no_grad():
        for rd in range(50):
            for d, q, r in cfgs:
                set_cfg(dec, d, q)
                x = xs[r]
                for _ in range(10):
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    net(x)
                    torch.cuda.synchronize()
                    times[key(d, q, r)].append((time.perf_counter() - t1) * 1000)
            if (rd + 1) % 10 == 0:
                log(f'  轮 {rd+1}/50  已用时 {time.time()-t0:.0f}s')

    results = load_json(TIMING_JSON)
    params_M = get_num_params(net) / 1e6
    log('[timing] 测 GFLOPs (thop, 逐配置按分辨率) ...')
    for d, q, r in cfgs:
        set_cfg(dec, d, q)
        a = np.array(times[key(d, q, r)])
        results[key(d, q, r)] = dict(
            depth=d, queries=q, resolution=r,
            median_ms=float(np.percentile(a, 50)),
            p95_ms=float(np.percentile(a, 95)),
            p99_ms=float(np.percentile(a, 99)),
            mean_ms=float(a.mean()),
            fps=float(1000.0 / np.percentile(a, 50)),
            gflops=float(get_flops(net, r)),
            params_M=float(params_M))
        save_json(results, TIMING_JSON)     # 每配置落盘, 断点续跑
        log(f"  d={d} q={q:<3} r={r}  P50={results[key(d,q,r)]['median_ms']:6.2f} "
            f"P95={results[key(d,q,r)]['p95_ms']:6.2f} "
            f"GFLOPs={results[key(d,q,r)]['gflops']:7.2f}")
    set_cfg(dec, 6, 300)
    log(f'[timing] 完成 -> {TIMING_JSON}')


# ============ 阶段 2/3 共用的 val 执行器 ============
def run_vals(configs, fraction, out_json, tag):
    from ultralytics import RTDETR
    m = RTDETR(MODEL_PATH)           # 单实例, 运行时改配置 (二维已验证)
    dec = find_decoder(m.model)
    results = load_json(out_json)
    for d, q, r in configs:
        k = key(d, q, r)
        if k in results and results[k].get('fraction') == fraction:
            log(f'[val {tag}] 跳过已完成 {k}')
            continue
        set_cfg(dec, d, q)
        t0 = time.time()
        mt = m.val(data=DATA_YAML, imgsz=r, batch=8, device=0, workers=3,
                   split='val', verbose=False, plots=False, fraction=fraction)
        results[k] = dict(depth=d, queries=q, resolution=r,
                          map50=float(mt.box.map50),
                          map50_95=float(mt.box.map),
                          fraction=fraction, source=tag)
        save_json(results, out_json)
        log(f'[val {tag}] {k}  mAP50={mt.box.map50:.4f} '
            f'mAP50-95={mt.box.map:.4f}  ({time.time()-t0:.0f}s)')
    set_cfg(dec, 6, 300)
    log(f'[val {tag}] 完成 -> {out_json}')


def stage_scan():
    cfgs = [(d, q, r) for d in DEPTHS for q in QUERIES for r in RESOLUTIONS]
    log(f'[scan] 27 配置 x fraction=0.2 ...')
    run_vals(cfgs, 0.2, SCAN_JSON, 'scan')


# ============ 阶段 3: 关键配置全量 val ============
def import_2d_full(configs):
    """r=640 且二维消融已有全量结果的配置: 直接导入 (同权重同协议)"""
    g2 = load_json(GRID_2D_JSON).get('results', {})
    imported, todo = {}, []
    for d, q, r in configs:
        if r == 640 and f'{d}_{q}' in g2 and 'map50_full' in g2[f'{d}_{q}']:
            g = g2[f'{d}_{q}']
            imported[key(d, q, r)] = dict(
                depth=d, queries=q, resolution=r,
                map50=g['map50_full'], map50_95=g['map_full'],
                fraction=1.0, source='full_2d')
        else:
            todo.append((d, q, r))
    return imported, todo


def auto_select_full():
    """用户锚点 ∪ 每 (depth,res) 扫描最优 queries 且 P50 <= 预算+1ms"""
    sel = set(DEFAULT_FULL)
    timing, scan = load_json(TIMING_JSON), load_json(SCAN_JSON)
    for d in DEPTHS:
        for r in RESOLUTIONS:
            cand = [(q, scan.get(key(d, q, r), {}).get('map50', -1))
                    for q in QUERIES]
            cand = [(q, v) for q, v in cand if v >= 0]
            if not cand:
                continue
            q_best = max(cand, key=lambda x: x[1])[0]
            t = timing.get(key(d, q_best, r), {}).get('median_ms', 1e9)
            if t <= BUDGET_MS + 1.0:
                sel.add((d, q_best, r))
    return sorted(sel, key=lambda c: (c[2], c[0], c[1]))[:9]


def stage_full(full_configs):
    if full_configs:
        configs = []
        for s in full_configs:
            d, q, r = s.split(',')
            configs.append((int(d), int(q), int(r)))
    else:
        configs = auto_select_full()
    log(f'[full] 目标配置 {len(configs)} 个: {configs}')

    results = load_json(FULL_JSON)
    imported, todo = import_2d_full(configs)
    new_imp = {k: v for k, v in imported.items() if k not in results}
    results.update(new_imp)
    if new_imp:
        save_json(results, FULL_JSON)
        log(f'[full] 从二维导入 {len(new_imp)} 个 r=640 全量结果: {list(new_imp)}')
    todo = [c for c in todo if key(*c) not in results]
    if todo:
        log(f'[full] 需实跑 {len(todo)} 个: {todo}')
        run_vals(todo, 1.0, FULL_JSON, 'full')
    else:
        log('[full] 无需实跑')


# ============ 阶段 4: 合并 ============
def stage_merge():
    timing, scan, full = load_json(TIMING_JSON), load_json(SCAN_JSON), load_json(FULL_JSON)
    head = ('depth,queries,resolution,map50,map50_95,median_ms,p95_ms,p99_ms,'
            'fps,gflops,eval_fraction,params_M,acc_source\n')
    lines = [head]
    rows = []
    for d in DEPTHS:
        for q in QUERIES:
            for r in RESOLUTIONS:
                k = key(d, q, r)
                t, acc = timing.get(k, {}), full.get(k, scan.get(k, {}))
                row = dict(depth=d, queries=q, resolution=r,
                           map50=acc.get('map50'), map50_95=acc.get('map50_95'),
                           median_ms=t.get('median_ms'), p95_ms=t.get('p95_ms'),
                           p99_ms=t.get('p99_ms'), fps=t.get('fps'),
                           gflops=t.get('gflops'), params_M=t.get('params_M'),
                           eval_fraction=acc.get('fraction'),
                           acc_source=acc.get('source', ''))
                rows.append(row)
                f = lambda v, nd: (f'{v:.{nd}f}' if isinstance(v, float)
                                   else ('' if v is None else v))
                lines.append(','.join([
                    str(d), str(q), str(r), f(row['map50'], 4), f(row['map50_95'], 4),
                    f(row['median_ms'], 2), f(row['p95_ms'], 2), f(row['p99_ms'], 2),
                    f(row['fps'], 1), f(row['gflops'], 2),
                    str(row['eval_fraction'] or ''), f(row['params_M'], 2),
                    row['acc_source']]) + '\n')
    with open(CSV_OUT, 'w', encoding='utf-8') as fo:
        fo.writelines(lines)
    log(f'[merge] -> {CSV_OUT}')

    # 15ms 预算前沿: 每 (depth,res) 取最优 queries, 看哪些进预算
    log('')
    log(f'=== {BUDGET_MS:.0f}ms 预算视角 (每 depth×res 的最优 queries) ===')
    log(f"{'d':>3}{'r':>6}{'best_q':>8}{'mAP50':>9}{'P50ms':>9}{'P95ms':>9}{'预算内?':>9}")
    for d in DEPTHS:
        for r in RESOLUTIONS:
            cand = [x for x in rows if x['depth'] == d and x['resolution'] == r
                    and x['map50'] is not None]
            if not cand:
                continue
            b = max(cand, key=lambda x: x['map50'])
            ok = '是' if b['median_ms'] and b['median_ms'] <= BUDGET_MS else '否'
            log(f"{d:>3}{r:>6}{b['queries']:>8}{b['map50']:>9.4f}"
                f"{b['median_ms']:>9.2f}{b['p95_ms']:>9.2f}{ok:>9}")


# ============ 主流程 ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', nargs='?', default='all',
                    choices=['timing', 'scan', 'full', 'merge', 'all'])
    ap.add_argument('--full-configs', nargs='+', default=[],
                    help='full 阶段手动指定, 形如 "3,375,512" "1,300,480"')
    args = ap.parse_args()

    if args.stage in ('timing', 'all'):
        log('阶段 1: 延迟测量 (~10 分钟)')
        stage_timing()
    if args.stage in ('scan', 'all'):
        log('阶段 2: 精度快扫 fraction=0.2 (~80 分钟)')
        stage_scan()
    if args.stage in ('full', 'all'):
        log('阶段 3: 关键配置全量 val')
        stage_full(args.full_configs)
    if args.stage in ('merge', 'all'):
        stage_merge()
    log('三维消融完成!')


if __name__ == '__main__':
    main()
