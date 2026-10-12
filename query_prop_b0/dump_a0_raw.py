# -*- coding: utf-8 -*-
"""dump_a0_raw.py —— 用指定 ultralytics 源码跑 6 帧 A0 推理并转储原始输出
用于回归测试: 原始代码 vs 修改后代码(temporal 关闭) 的行为一致性。
用法: python dump_a0_raw.py <ultralytics_repo_root> <out.npz>
"""
import os
import sys

import numpy as np
from PIL import Image

repo = os.path.abspath(sys.argv[1])
out = sys.argv[2]
sys.path.insert(0, repo)

import ultralytics  # noqa: E402
assert ultralytics.__file__.lower().startswith(repo.lower()), \
    f'导入路径错误: {ultralytics.__file__} != {repo}'
from ultralytics import RTDETR  # noqa: E402

DATA = r'D:\lunwen\自研\数据集\argoverse-hd'
CKPT = r'D:\lunwen\自研\runs\detect\runs_rtdetr\r18_640_sub20_b4\weights\best.pt'

import json  # noqa: E402

with open(os.path.join(DATA, 'stream_index.json'), encoding='utf-8') as f:
    idx = json.load(f)
files = [os.path.join(DATA, 'images', fr['file']) for fr in idx['scenes'][0]['frames'][:6]]

m = RTDETR(CKPT)
head = None
for mod in m.model.modules():
    if type(mod).__name__ == 'RTDETRDecoder' and hasattr(mod, 'decoder'):
        head = mod
        break
head.decoder.eval_idx = 2

raws, dets = [], []
for p in files:
    with Image.open(p) as im:
        bgr = np.array(im.convert('RGB'))[:, :, ::-1].copy()
    r = m.predict(bgr, imgsz=640, conf=0.001, max_det=300, device=0, verbose=False)[0]
    b = r.boxes
    d = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                        b.cls.cpu().numpy()[:, None]], 1) if len(b) else np.zeros((0, 6))
    dets.append(d.astype(np.float32))
    # 原始 postprocess 输出 (topk 后、conf 过滤前), 修改版通过 meta 也可取到分数,
    # 这里直接用 predictor 输入侧的 y: 从 model 输出拿不到就退化为检测结果
    raws.append(d.astype(np.float32))

np.savez(out, **{f'frame{i}': dets[i] for i in range(len(dets))})
print('已写入', out, '源码:', ultralytics.__file__)
