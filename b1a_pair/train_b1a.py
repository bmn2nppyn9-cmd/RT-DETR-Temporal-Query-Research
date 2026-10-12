# -*- coding: utf-8 -*-
"""
train_b1a.py —— B1a 五配置短训驱动
=====================================
矩阵(第一轮 M=25, 全部从 r18_640_sub20_b4/best.pt 初始化, 统一 seed/步数/超参):
  b1a0    3L/300 current        单图  训练基线
  b1ac25  3L/325 current        单图  等 query 数对照(eval_topk=300)
  b1at31  3L/300+25 S3→L1 前    pair  深度错位对照
  b1at23  3L/300+25 S2→L3 前    pair  候选
  b1at33  3L/300+25 S3→L3 前    pair  候选

用法:
  python train_b1a.py --configs b1a0 b1ac25 --epochs 1          # smoke
  python train_b1a.py --epochs 3                                # 全矩阵主实验
  python train_b1a.py --configs b1at33 --epochs 1 --skip 1      # 续跑(跳过已完成)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqinfra.runtime import activate_backend, DATA_ROOT, PAIR_DIR, TRAIN_DIR, RECORD_DIR, EVAL_DIR, INITIAL_WEIGHTS, MODEL_YAML, ROOT
activate_backend()
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

HERE = str(PAIR_DIR)
PAIR_DIR.mkdir(parents=True, exist_ok=True)
BEST_PT = str(INITIAL_WEIGHTS)
PROJECT = str(TRAIN_DIR)

CONFIGS = {
    'b1a0': {
        'struct': {'ndl': 3, 'nq': 300, 'nd': 0},
        'pair': False, 'desc': '3L/300 current 训练基线',
    },
    'b1ac25': {
        'struct': {'ndl': 3, 'nq': 325, 'nd': 0, 'eval_topk': 300},
        'pair': False, 'desc': '3L/325 current 等 query 数对照',
    },
    'b1at31': {
        'struct': {'ndl': 3, 'nq': 300, 'nd': 0,
                   'temporal': {'M': 25, 'source_depth': 3, 'injection_layer': 0}},
        'pair': True, 'desc': 'S3→L1 前 深度错位对照',
    },
    'b1at23': {
        'struct': {'ndl': 3, 'nq': 300, 'nd': 0,
                   'temporal': {'M': 25, 'source_depth': 2, 'injection_layer': 2}},
        'pair': True, 'desc': 'S2→L3 前 候选',
    },
    'b1at33': {
        'struct': {'ndl': 3, 'nq': 300, 'nd': 0,
                   'temporal': {'M': 25, 'source_depth': 3, 'injection_layer': 2}},
        'pair': True, 'desc': 'S3→L3 前 候选',
    },
}

DATA_YAML = {
    False: os.path.join(HERE, 'b1a_single.yaml'),
    True: os.path.join(HERE, 'b1a_pair.yaml'),
}


def write_yamls():
    head = ('# 自动生成 by train_b1a.py\n'
            'path: {data_root}\n'
            'val: {val}\n'
            'names:\n'
            '  0: person\n  1: bicycle\n  2: car\n  3: motorcycle\n'
            '  4: bus\n  5: truck\n  6: traffic_light\n  7: stop_sign\n')
    val = os.path.join(HERE, 'val_sub20.txt').replace('\\', '/')
    for pair, txt in [(False, 't_frames_train.txt'), (True, 'all_frames_train.txt')]:
        p = DATA_YAML[pair]
        with open(p, 'w', encoding='utf-8') as f:
            f.write(f'train: {json.dumps(os.path.join(HERE, txt).replace(chr(92), "/"))}\n'
                    + head.format(val=json.dumps(val), data_root=json.dumps(str(DATA_ROOT / 'argoverse-hd-yolo').replace(chr(92), '/'))))


def run_config(name: str, epochs: int, batch=4, device='0', workers=3) -> dict:
    from dataset_b1a import B1aTrainer  # 延迟导入, 保证 override 生效时机

    cfg = CONFIGS[name]
    pair_index = None
    if cfg['pair']:
        pair_index = json.load(open(os.path.join(HERE, 'pairs_train.json'),
                                    encoding='utf-8'))
    overrides = dict(
        model=str(MODEL_YAML),
        data=DATA_YAML[cfg['pair']],
        epochs=epochs,
        batch=batch,
        imgsz=640,
        device=device,
        workers=workers,
        amp=False,
        deterministic=False,
        seed=0,
        optimizer='AdamW',
        lr0=1e-4,
        lrf=0.1,
        warmup_epochs=0.0,          # 微调收敛模型: 禁止 warmup
        warmup_bias_lr=1e-4,        # 默认 0.1 会在 warmup 期摧毁分类头偏置(已实证)
        pretrained=BEST_PT,
        project=PROJECT,
        name=name,
        exist_ok=False,
        val=True,
        save=True,
        verbose=True,
    )
    t0 = time.time()
    trainer = B1aTrainer(overrides=overrides, b1a_struct=cfg['struct'],
                         pair_index=pair_index)
    trainer.train()
    dt = time.time() - t0

    # 汇总: validator 指标是 A0 口径(temporal 关闭)的健康指标, 不是 B1a 判据
    res = {
        'config': name, 'desc': cfg['desc'], 'struct': cfg['struct'],
        'pair': cfg['pair'], 'epochs': epochs, 'train_minutes': round(dt / 60, 1),
        'best_pt': str(trainer.best), 'last_pt': str(trainer.last),
    }
    try:
        m = trainer.validator.metrics
        res['val_a0_metrics'] = {
            'map50': float(m.box.map50), 'map': float(m.box.map),
            'note': 'A0 口径(temporal 关闭), 仅作健康检查, 非 B1a 判据',
        }
    except Exception as e:
        res['val_a0_metrics'] = {'error': str(e)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', choices=list(CONFIGS), default=list(CONFIGS))
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--device', default='0')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    write_yamls()
    os.makedirs(PROJECT, exist_ok=True)
    results = []
    for name in args.configs:
        print(f'\n{"=" * 70}\n[B1a] {name}: {CONFIGS[name]["desc"]}  '
              f'epochs={args.epochs}\n{"=" * 70}', flush=True)
        try:
            r = run_config(name, args.epochs, args.batch, args.device, args.workers)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = {'config': name, 'error': str(e)}
        results.append(r)
        print(f'[B1a] {name} 完成: {r.get("train_minutes", "?")} min, '
              f'best={r.get("best_pt", r.get("error", "?"))}', flush=True)
        # 每组落盘一次, 防中途崩溃丢结果
        out = args.out or os.path.join(HERE, f'b1a_smoke_e{args.epochs}_results.json')
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\n[B1a] 全部完成, 结果: {out}')
    if any('error' in r for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
