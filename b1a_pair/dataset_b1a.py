# -*- coding: utf-8 -*-
"""
dataset_b1a.py —— B1a pair 数据集与训练器
==============================================
PairRTDETRDataset: 每个样本 = (t-1 帧图像, t 帧图像+标签)。
  - t 帧走标准 RTDETRDataset 标签管线(augment=False, 仅 LetterBox+Format);
  - t-1 帧经同一 transforms 管线(几何一致), 其标签被忽略(状态提取不需要 GT);
  - collate_fn 增加 img_prev 的 stack(标准 collate 只 stack img)。

B1aTrainer: 在标准 RTDETRTrainer 上挂三件小事:
  - 构造前设置 RTDETRDecoder.b1a_struct_override(结构覆盖);
  - build_dataset 在 train 模式且给定了 pair_index 时返回 PairRTDETRDataset;
  - preprocess_batch 增加 img_prev 的 /255(标准实现只对 img 做)。
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqinfra.runtime import activate_backend, DATA_ROOT, PAIR_DIR, TRAIN_DIR, RECORD_DIR, EVAL_DIR, INITIAL_WEIGHTS, MODEL_YAML, ROOT
activate_backend()

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.val import RTDETRDataset
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.utils import colorstr

HERE = os.path.dirname(os.path.abspath(__file__))


class PairRTDETRDataset(RTDETRDataset):
    """(t-1, t) 帧对数据集。im_files 为 t 与 t-1 的并集清单。"""

    def __init__(self, *args, pair_index=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pairs = pair_index  # list[(prev_path, cur_path)]
        self._file2idx = {os.path.normpath(f): i for i, f in enumerate(self.im_files)}
        miss = [p for pair in self.pairs for p in pair
                if os.path.normpath(p) not in self._file2idx]
        if miss:
            raise RuntimeError(f'pair 中有 {len(miss)} 个文件不在 im_files, 示例: {miss[:2]}')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        prev_path, cur_path = self.pairs[index]
        label = super().__getitem__(self._file2idx[os.path.normpath(cur_path)])
        prev = self.get_image_and_label(self._file2idx[os.path.normpath(prev_path)])
        prev = self.transforms(prev)  # 同一管线保证几何一致; 标签忽略
        label['img_prev'] = prev['img']
        return label

    @staticmethod
    def collate_fn(batch):
        new_batch = RTDETRDataset.collate_fn(batch)
        new_batch['img_prev'] = torch.stack([b['img_prev'] for b in batch], 0)
        return new_batch


class B1aTrainer(RTDETRTrainer):
    """B1a 训练器: 结构覆盖 + pair 数据集 + img_prev 归一化。"""

    def __init__(self, overrides, b1a_struct: dict | None = None,
                 pair_index: list | None = None, _callbacks=None):
        RTDETRDecoder.b1a_struct_override = b1a_struct  # 必须先于 get_model
        self._b1a_pair_index = pair_index
        super().__init__(overrides=overrides, _callbacks=_callbacks)

    def build_dataset(self, img_path, mode='train', batch=None):
        if mode == 'train' and self._b1a_pair_index is not None:
            return PairRTDETRDataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=False,               # 首轮关闭全部随机几何增强(pair 对应性)
                hyp=self.args,
                rect=False,
                cache=self.args.cache or None,
                single_cls=self.args.single_cls or False,
                prefix=colorstr(f'{mode}: '),
                classes=self.args.classes,
                data=self.data,
                pair_index=self._b1a_pair_index,
            )
        return super().build_dataset(img_path, mode, batch)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if isinstance(batch.get('img_prev'), torch.Tensor):
            batch['img_prev'] = batch['img_prev'].float() / 255
        return batch
