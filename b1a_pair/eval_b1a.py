# -*- coding: utf-8 -*-
"""
eval_b1a.py —— B1a 评测与 Go/No-Go 判定驱动
==============================================
输入: records 目录(record_b1a.py 产出的 preds_b1a*.npz)
步骤:
  1. preds_b1a0.npz 复制为 preds_a0.npz(eval_qd_t 的参照命名);
  2. eval_full_metrics.py 出全指标(AP/AP50/AP75/尺寸/AR/工作点);
  3. eval_qd_t.py 出拆源诊断(current-only/temporal-only/保持率/新增 TP);
  4. 按任务书第 8 节判 Go/No-Go, 写 b1a_decision.json。

用法:
  python eval_b1a.py --records records_e1 --suffix e1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqinfra.runtime import EVAL_DIR, RECORD_DIR, ROOT
EVAL_DIR.mkdir(parents=True, exist_ok=True)
HERE = str(EVAL_DIR)
ZIYAN = str(ROOT)
TAGS = ['b1a0', 'b1ac25', 'b1at31', 'b1at23', 'b1at33']

# 历史锚点(冻结模型, 仅对照语境): A0=3L 冻结, U0=6L 冻结
ANCHOR = {'A0_frozen': 0.38864, 'U0_frozen': 0.40740}


def run(cmd, logm):
    print(f'[eval_b1a] {logm}', flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', default=str(RECORD_DIR))
    ap.add_argument('--suffix', default='e1')
    ap.add_argument('--tags', nargs='+', default=TAGS)
    ap.add_argument('--skip-full', action='store_true')
    ap.add_argument('--skip-split', action='store_true')
    args = ap.parse_args()

    rec = args.records
    have = [t for t in args.tags if os.path.exists(os.path.join(rec, f'preds_{t}.npz'))]
    missing = [t for t in args.tags if t not in have]
    if missing:
        raise FileNotFoundError(f'Missing recordings: {missing}')
    assert 'b1a0' in have, '缺 preds_b1a0.npz(基线), 无法判定'

    a0_ref = os.path.join(rec, 'preds_a0.npz')
    if not os.path.exists(a0_ref):
        shutil.copy(os.path.join(rec, 'preds_b1a0.npz'), a0_ref)
        print('[eval_b1a] preds_b1a0.npz -> preds_a0.npz(拆源参照)')

    full_out = os.path.join(HERE, f'out_full_metrics_b1a_{args.suffix}.json')
    if not args.skip_full:
        run([sys.executable, os.path.join(ZIYAN, 'eval_full_metrics.py'),
             '--preds-dir', rec, '--tags', *have, '--out', full_out],
            '全指标评测(AP/AP50/AP75/尺寸/AR/工作点)...')

    split_out = os.path.join(HERE, f'temporal_b1a_{args.suffix}.json')
    if not args.skip_split:
        run([sys.executable, os.path.join(ZIYAN, 'query_depth_probe', 'eval_qd_t.py'),
             '--preds-dir', rec, '--configs', *have, '--out', split_out],
            '拆源诊断(current/temporal/保持率/新增 TP)...')

    full = json.load(open(full_out, encoding='utf-8'))
    split = json.load(open(split_out, encoding='utf-8'))

    # ---------- 汇总表 ----------
    rows = {}
    for t in have:
        f = full.get(t, {})
        s = split.get(t, {})
        rows[t] = {
            'AP50': f.get('AP50'), 'AP': f.get('AP'), 'AP75': f.get('AP75'),
            'AP_small': f.get('AP_small'), 'AR100': f.get('AR100'),
            'current_only_map50': (s.get('ap_current_only') or {}).get('map50'),
            'temporal_only_map50': (s.get('ap_temporal_only') or {}).get('map50'),
            'keep95': (s.get('current_keep_vs_a0') or {}).get('iou0.95'),
            'new_gt_temporal': s.get('new_gt_temporal_only'),
            'a0_gt_lost': s.get('a0_gt_lost'),
        }

    print('\n' + '=' * 96)
    hdr = f"{'配置':<9}{'AP50':>8}{'AP':>8}{'AP75':>8}{'AP_s':>7}{'AR100':>7}" \
          f"{'cur50':>8}{'tmp50':>8}{'keep95':>8}{'newT':>7}{'lost':>7}"
    print(hdr)
    for t in have:
        r = rows[t]
        fmt = lambda v, w=8, d=4: (f'{v:>{w}.{d}f}' if isinstance(v, float) else f'{str(v):>{w}}')
        print(f"{t:<9}{fmt(r['AP50'])}{fmt(r['AP'])}{fmt(r['AP75'])}"
              f"{fmt(r['AP_small'], 7)}{fmt(r['AR100'], 7)}"
              f"{fmt(r['current_only_map50'])}{fmt(r['temporal_only_map50'])}"
              f"{fmt(r['keep95'], 8, 3)}{fmt(r['new_gt_temporal'], 7)}{fmt(r['a0_gt_lost'], 7)}")
    print(f"锚点(冻结模型): A0={ANCHOR['A0_frozen']}  U0={ANCHOR['U0_frozen']}")
    print('=' * 96)

    # ---------- Go/No-Go(任务书第 8 节) ----------
    checks = []
    base = rows['b1a0']
    cands = [t for t in ('b1at23', 'b1at33') if t in rows]
    if cands:
        best = max(cands, key=lambda t: rows[t]['AP50'])
        br = rows[best]
        c1 = (br['AP50'] - base['AP50'] >= 0.005) or (br['AP'] - base['AP'] >= 0.003)
        checks.append(('late injection 相对 B1a-0: mAP50 +0.5pp 或 AP +0.3pp', c1,
                       f"{best}: AP50 {br['AP50'] - base['AP50']:+.4f}, AP {br['AP'] - base['AP']:+.4f}"))
        if 'b1ac25' in rows:
            c2 = br['AP50'] > rows['b1ac25']['AP50']
            checks.append(('优于等 query 数 current control', c2,
                           f"{best} {br['AP50']:.4f} vs b1ac25 {rows['b1ac25']['AP50']:.4f}"))
        if br['current_only_map50'] is not None:
            drop = base['AP50'] - br['current_only_map50']
            c3 = drop <= 0.003
            checks.append(('current-only 相对 B1a-0 下降 ≤0.3pp', c3,
                           f'降 {drop:+.4f}'))
        if False:  # Frozen cross-model TP criterion is invalid for independently trained models.
            c4 = br['new_gt_temporal'] > br['a0_gt_lost']
            checks.append(('temporal 新增 TP > current 损失', c4,
                           f"new {br['new_gt_temporal']} vs lost {br['a0_gt_lost']}"))
        if 'b1at31' in rows:
            c5 = br['AP50'] > rows['b1at31']['AP50']
            checks.append(('T23/T33 至少一个优于 T31', c5,
                           f"{best} {br['AP50']:.4f} vs b1at31 {rows['b1at31']['AP50']:.4f}"))

    print('\nGo/No-Go 判定:')
    n_pass = 0
    for name, ok, detail in checks:
        n_pass += bool(ok)
        print(f"  [{'通过' if ok else '未过'}] {name}  ({detail})")
    verdict = ('Go' if n_pass == len(checks) and checks else
               'No-Go' if n_pass <= len(checks) // 2 else '边界, 需人工判读')
    print(f'\n结论: {verdict}  ({n_pass}/{len(checks)})')
    print('注意: 本表为 smoke(1 epoch)趋势时不作最终结论, 报告须标注训练轮数。')

    out = {'suffix': args.suffix, 'records': rec, 'rows': rows,
           'checks': [{'name': n, 'pass': bool(o), 'detail': d} for n, o, d in checks],
           'verdict': verdict, 'anchor': ANCHOR, 'scope': 'same-frame AP; not an sAP or production release gate'}
    out_path = os.path.join(HERE, f'b1a_decision_{args.suffix}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f'已写入 {out_path}')


if __name__ == '__main__':
    main()
