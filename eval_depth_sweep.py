# -*- coding: utf-8 -*-
"""
eval_depth_sweep.py —— 解码器层数 vs 精度的折中曲线(可行性验证用)
==================================================================
目的: 回答"少跑 decoder 层, 精度掉多少"
     给出 层数 -> (mAP50, mAP50-95, 延迟) 的对照表, 用于证明
     "省时间的同时精度不下降(或下降在可接受范围内)"。

做法: 直接改写 DeformableTransformerDecoder.eval_idx 截断层数,
      跑 ultralytics 的 val() 得到官方口径指标。

用法:
    python eval_depth_sweep.py --depths 1 2 3 4 5 6 --fraction 0.1
    python eval_depth_sweep.py --depths 1 3 6          # 全量 val
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def log(m): print(m, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=r'runs/detect/runs_rtdetr/r18_640_sub20_b4/weights/best.pt')
    ap.add_argument('--data', default=r'cfg/argoverse-hd-local.yaml')
    ap.add_argument('--depths', nargs='+', type=int, default=[1,2,3,4,5,6])
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--fraction', type=float, default=1.0,
                    help='验证集使用比例(1.0=全量 val)。ultralytics 的 val 不接受 limit 参数')
    ap.add_argument('--out', default='out_depth_sweep.json')
    args = ap.parse_args()

    from ultralytics import RTDETR
    m = RTDETR(args.model)
    inner = None
    for mod in m.model.modules():
        if type(mod).__name__ == 'RTDETRDecoder' and hasattr(mod, 'decoder'):
            inner = mod.decoder; break
    if inner is None:
        log('[错误] 找不到 DeformableTransformerDecoder'); sys.exit(1)
    log(f'decoder: num_layers={inner.num_layers}  eval_idx={inner.eval_idx}')
    full = inner.num_layers
    saved = inner.eval_idx

    results = []
    log('')
    log(f"{'层数':<6}{'mAP50':>10}{'mAP50-95':>12}{'Precision':>12}{'Recall':>10}{'延迟(ms)':>12}")
    for d in args.depths:
        inner.eval_idx = d - 1
        r = m.val(data=args.data, imgsz=args.imgsz, batch=args.batch,
                  device=0, workers=3, split='val', verbose=False,
                  plots=False, fraction=args.fraction)
        b = r.box
        # 单独测纯推理延迟(合成输入, 中位数)
        x = torch.rand(1, 3, args.imgsz, args.imgsz).cuda()
        mm = m.model.cuda().eval()
        with torch.no_grad():
            for _ in range(3): mm(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(20):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad(): mm(x)
            torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1000)
        lat = float(np.median(ts))
        results.append(dict(depth=d, map50=float(b.map50), map=b.map,
                            precision=float(b.mp), recall=float(b.mr),
                            latency_ms=lat))
        log(f"{d:<6}{b.map50:>10.4f}{b.map:>12.4f}{b.mp:>12.4f}{b.mr:>10.4f}{lat:>12.2f}")

    inner.eval_idx = saved
    ref = next((x for x in results if x['depth'] == full), None)
    if ref:
        log('')
        log(f'=== 相对 {full} 层的差距 ===')
        log(f"{'层数':<6}{'ΔmAP50(pp)':>14}{'ΔmAP50-95(pp)':>16}{'省时(%)':>10}")
        for x in results:
            log(f"{x['depth']:<6}{(x['map50']-ref['map50'])*100:>+14.2f}"
                f"{(x['map']-ref['map'])*100:>+16.2f}"
                f"{(1-x['latency_ms']/ref['latency_ms'])*100:>10.1f}")
    with open(args.out,'w',encoding='utf-8') as f:
        json.dump(dict(fraction=args.fraction, results=results), f, ensure_ascii=False, indent=2)
    log('')
    log(f'已写入 {args.out}')

if __name__ == '__main__':
    main()
