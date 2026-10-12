# -*- coding: utf-8 -*-
"""
eval_grid_2d.py —— RT-DETR 二维消融网格: Decoder层数 × Query数
================================================================
同一权重 (r18_640_sub20_b4/best.pt), 纯推理期配置调整, 不重训。

旋钮 (均为运行时实例属性, 源码已核实):
  层数: RTDETRDecoder.decoder.eval_idx = d-1
        (transformer.py:792 推理时 i==eval_idx 即 break, 真提前退出)
  查询: RTDETRDecoder.num_queries = q
        (head.py:1709 encoder topk 选查询; head.py:1606 输出 topk;
         分类/回归头是逐 query MLP, 与 query 数无关)

阶段:
  timing : 30 配置交错测延迟 —— 每配置 50 预热 + 500 计时
           (50 轮 × 每轮每配置 10 次, 轮转交错排除热漂移), 另测 GFLOPs
  scan   : 30 配置 × fraction=0.2 (3012 帧) 子集快扫精度
  full   : 指定配置全量 val, 例:
           python eval_grid_2d.py --phase full --full-configs "1,300" "3,150"

每配置跑完即写 out_grid_2d.json + ablation_2d_results.csv (断点续跑)。
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
os.chdir(ROOT)                           # 沿用既有脚本的相对路径习惯

DEPTHS = [1, 2, 3, 4, 5, 6]
QUERIES = [75, 150, 225, 300, 375]
SEED = 20260916

JSON_OUT = os.path.join(WORKDIR, 'out_grid_2d.json')
CSV_OUT = os.path.join(WORKDIR, 'ablation_2d_results.csv')

MODEL = r'runs/detect/runs_rtdetr/r18_640_sub20_b4/weights/best.pt'
DATA = r'cfg/argoverse-hd-local.yaml'


def log(m):
    print(m, flush=True)


def find_decoder(model):
    for mod in model.modules():
        if type(mod).__name__ == 'RTDETRDecoder' and hasattr(mod, 'num_queries'):
            return mod
    return None


def set_cfg(dec, d, q):
    dec.decoder.eval_idx = d - 1
    dec.num_queries = q


def key(d, q):
    return f'{d}_{q}'


def load_data():
    if os.path.exists(JSON_OUT):
        with open(JSON_OUT, encoding='utf-8') as f:
            return json.load(f)
    return {
        'meta': dict(model=MODEL, data=DATA, seed=SEED, imgsz=640,
                     val_batch=8, scan_fraction=0.2,
                     timing=dict(warmup=50, rounds=50, iters_per_round=10,
                                 total_per_cfg=500, interleaved=True,
                                 input='torch.rand(1,3,640,640) seeded',
                                 fps_definition='1000/median'),
                     depths=DEPTHS, queries=QUERIES),
        'results': {},
    }


def save(data):
    tmp = JSON_OUT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, JSON_OUT)
    write_csv(data)


def write_csv(data):
    head = ('depth,queries,map50,map50_95,median_ms,p95_ms,p99_ms,fps,gflops,'
            'eval_fraction,params_M,'
            'median_ms_fused,p95_ms_fused,p99_ms_fused,fps_fused,gflops_fused,params_M_fused\n')
    lines = [head]
    for d in DEPTHS:
        for q in QUERIES:
            r = data['results'].get(key(d, q), {})
            full = 'map50_full' in r
            map50 = r.get('map50_full' if full else 'map50_sub', '')
            map95 = r.get('map_full' if full else 'map_sub', '')
            frac = '1.0' if full else ('0.2' if 'map50_sub' in r else '')
            def f(v, nd=4):
                return f'{v:.{nd}f}' if isinstance(v, float) else (v if v != '' else '')
            lines.append(','.join([
                str(d), str(q), f(map50), f(map95),
                f(r.get('p50_ms', ''), 2), f(r.get('p95_ms', ''), 2),
                f(r.get('p99_ms', ''), 2), f(r.get('fps', ''), 1),
                f(r.get('gflops', ''), 2), frac, f(r.get('params_M', ''), 2),
                f(r.get('p50_ms_f', ''), 2), f(r.get('p95_ms_f', ''), 2),
                f(r.get('p99_ms_f', ''), 2), f(r.get('fps_f', ''), 1),
                f(r.get('gflops_f', ''), 2), f(r.get('params_M_f', ''), 2),
            ]) + '\n')
    with open(CSV_OUT, 'w', encoding='utf-8') as fo:
        fo.writelines(lines)


def phase_timing(m, dec, data, fuse=False):
    import torch
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    sfx = '_f' if fuse else ''
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    net = m.model.cuda().eval()
    if fuse:
        net.fuse()
        log(f'[timing] 已 fuse: params={get_num_params(net)/1e6:.2f}M')
    x = torch.rand(1, 3, 640, 640, device='cuda')
    cfgs = [(d, q) for d in DEPTHS for q in QUERIES]
    times = {key(d, q): [] for d, q in cfgs}

    # 预热: 每配置 50 次
    log('[timing] 预热 每配置50次 ...')
    with torch.no_grad():
        for d, q in cfgs:
            set_cfg(dec, d, q)
            for _ in range(50):
                net(x)
    torch.cuda.synchronize()

    # 计时: 50 轮 × 每轮每配置 10 次 (轮转交错)
    log('[timing] 计时 50轮 x 30配置 x 10次 = 每配置500次 ...')
    t_start = time.time()
    with torch.no_grad():
        for rd in range(50):
            for d, q in cfgs:
                set_cfg(dec, d, q)
                for _ in range(10):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    net(x)
                    torch.cuda.synchronize()
                    times[key(d, q)].append((time.perf_counter() - t0) * 1000)
            if (rd + 1) % 10 == 0:
                log(f'  轮 {rd+1}/50  已用时 {time.time()-t_start:.0f}s')

    params_M = get_num_params(net) / 1e6
    log('[timing] 测 GFLOPs (thop, 逐配置) ...')
    for d, q in cfgs:
        set_cfg(dec, d, q)
        ts = np.array(times[key(d, q)])
        r = data['results'].setdefault(key(d, q), {})
        r.update({f'p50_ms{sfx}': float(np.percentile(ts, 50)),
                  f'p95_ms{sfx}': float(np.percentile(ts, 95)),
                  f'p99_ms{sfx}': float(np.percentile(ts, 99)),
                  f'mean_ms{sfx}': float(ts.mean()),
                  f'fps{sfx}': float(1000.0 / np.percentile(ts, 50)),
                  f'gflops{sfx}': float(get_flops(net, 640)),
                  f'params_M{sfx}': float(params_M)})
        save(data)
        log(f"  d={d} q={q:<4} P50={r[f'p50_ms{sfx}']:6.2f} P95={r[f'p95_ms{sfx}']:6.2f} "
            f"P99={r[f'p99_ms{sfx}']:6.2f} FPS={r[f'fps{sfx}']:6.1f} GFLOPs={r[f'gflops{sfx}']:7.2f}")
    set_cfg(dec, 6, 300)


def run_vals(m, dec, data, cfgs, fraction, tag):
    for d, q in cfgs:
        set_cfg(dec, d, q)
        t0 = time.time()
        r = m.val(data=DATA, imgsz=640, batch=8, device=0, workers=3,
                  split='val', verbose=False, plots=False, fraction=fraction)
        rec = data['results'].setdefault(key(d, q), {})
        rec[f'map50_{tag}'] = float(r.box.map50)
        rec[f'map_{tag}'] = float(r.box.map)
        save(data)
        log(f'[val {tag}] d={d} q={q:<4} mAP50={r.box.map50:.4f} '
            f'mAP50-95={r.box.map:.4f}  ({time.time()-t0:.0f}s)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='timing,scan',
                    help='逗号分隔: timing,scan,full')
    ap.add_argument('--full-configs', nargs='+', default=[],
                    help='full 阶段配置, 形如 "1,300" "3,150"')
    ap.add_argument('--fuse', action='store_true',
                    help='timing 前先 fuse (BN折叠), 对应部署/历史口径')
    args = ap.parse_args()

    from ultralytics import RTDETR
    m = RTDETR(MODEL)
    dec = find_decoder(m.model)
    if dec is None:
        log('[错误] 找不到 RTDETRDecoder')
        sys.exit(1)
    log(f'RTDETRDecoder: num_queries={dec.num_queries} '
        f'num_layers={dec.decoder.num_layers} eval_idx={dec.decoder.eval_idx}')

    data = load_data()
    phases = args.phase.split(',')

    if 'timing' in phases:
        phase_timing(m, dec, data, fuse=args.fuse)
    if 'scan' in phases:
        cfgs = [(d, q) for d in DEPTHS for q in QUERIES]
        log(f'[scan] 子集快扫 fraction=0.2, {len(cfgs)} 配置 ...')
        run_vals(m, dec, data, cfgs, 0.2, 'sub')
    if 'full' in phases:
        cfgs = []
        for s in args.full_configs:
            d, q = s.split(',')
            cfgs.append((int(d), int(q)))
        log(f'[full] 全量 val, {len(cfgs)} 配置 ...')
        run_vals(m, dec, data, cfgs, 1.0, 'full')

    log(f'完成. JSON={JSON_OUT}  CSV={CSV_OUT}')


if __name__ == '__main__':
    main()
