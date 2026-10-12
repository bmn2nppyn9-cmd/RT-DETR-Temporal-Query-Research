# -*- coding: utf-8 -*-
"""
test_b1a_regression.py —— B1a 训练前回归门(R1-R10)
====================================================
全过才允许开训。对应 B1a_实现方案.md 第 5 节。
用法:
    cd D:\lunwen\自研\b1a_pair
    python test_b1a_regression.py
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqinfra.runtime import activate_backend, DATA_ROOT, PAIR_DIR, TRAIN_DIR, RECORD_DIR, EVAL_DIR, INITIAL_WEIGHTS, MODEL_YAML, ROOT
activate_backend()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.nn.tasks import RTDETRDetectionModel, torch_load
from ultralytics.utils import LOGGER

HERE = str(PAIR_DIR)
BEST_PT = str(INITIAL_WEIGHTS)
YAML = str(MODEL_YAML)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NC = 8

PASS, FAIL = 'PASS', 'FAIL'
results = []


def report(rid, ok, detail):
    results.append((rid, ok, detail))
    LOGGER.info(f'[{PASS if ok else FAIL}] {rid}: {detail}')


def build(struct):
    RTDETRDecoder.b1a_struct_override = struct
    m = RTDETRDetectionModel(YAML, ch=3, nc=NC, verbose=False)
    RTDETRDecoder.b1a_struct_override = None
    m.nc = NC  # 正式流程由 trainer 挂载(detect/train.py); 测试手动补齐
    return m


def load_weights(m):
    ckpt = torch_load(BEST_PT, weight_only=False) if False else torch.load(
        BEST_PT, map_location='cpu', weights_only=False)
    m.load(ckpt)
    return m


def make_batch(n=2):
    """从 pair 数据集取真实 batch(含 img_prev/标签), 搬到设备并复刻 trainer 预处理"""
    from dataset_b1a import PairRTDETRDataset
    pairs = json.load(open(os.path.join(HERE, 'pairs_train.json'), encoding='utf-8'))[:n]
    hyp = SimpleNamespace(mask_ratio=4, overlap_mask=True, bgr=0.0)
    ds = PairRTDETRDataset(
        img_path=os.path.join(HERE, 'all_frames_train.txt'), imgsz=640, batch_size=n,
        augment=False, hyp=hyp, rect=False, cache=False, single_cls=False,
        prefix='', classes=None, data={'channels': 3, 'nc': NC,
                                       'names': {i: str(i) for i in range(NC)}},
        pair_index=pairs)
    batch = PairRTDETRDataset.collate_fn([ds[i] for i in range(n)])
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(DEVICE)
    batch['img'] = batch['img'].float() / 255
    batch['img_prev'] = batch['img_prev'].float() / 255
    return batch


def main():
    torch.manual_seed(0)

    # ---------- R1: ndl=3 从 6 层 ckpt 加载 ≡ 6 层模型 eval_idx=2(逐位)
    m3 = load_weights(build({'ndl': 3, 'nq': 300, 'nd': 0})).to(DEVICE).eval()
    m6 = load_weights(build(None)).to(DEVICE).eval()
    m6.model[-1].decoder.eval_idx = 2
    batch = make_batch(2)
    img = batch['img']
    with torch.no_grad():
        y3 = m3(img)
        y6 = m6(img)
    o3, o6 = y3[0], y6[0]
    diff = (o3 - o6).abs().max().item()
    report('R1 ndl=3 加载与结构', diff == 0.0,
           f'3 层模型 vs 6 层模型 eval_idx=2 输出最大差 {diff}（应逐位一致）')

    # ---------- R2: pair 数据集(同场景/几何/collate)
    from dataset_b1a import PairRTDETRDataset
    pairs = json.load(open(os.path.join(HERE, 'pairs_train.json'), encoding='utf-8'))
    same_scene = all(os.path.basename(p).split('_315')[0] ==
                     os.path.basename(c).split('_315')[0] for p, c in pairs[:200])
    shapes_ok = batch['img'].shape == batch['img_prev'].shape == (2, 3, 640, 640)
    report('R2 pair 数据集', same_scene and shapes_ok,
           f'前 200 对同场景={same_scene}, img/img_prev 形状={tuple(batch["img"].shape)}')

    # ---------- R3: M=0(state=None) pair 路径 ≡ 对照路径(损失逐位)
    mc = load_weights(build({'ndl': 3, 'nq': 300, 'nd': 0})).to(DEVICE).train()
    mt = load_weights(build({'ndl': 3, 'nq': 300, 'nd': 0,
                             'temporal': {'M': 25, 'source_depth': 3,
                                          'injection_layer': 2}})).to(DEVICE).train()
    batch_c = {k: v for k, v in batch.items() if k != 'img_prev'}
    with torch.no_grad():
        lc, _ = mc.loss(batch_c)
        lt, _ = mt.loss(batch_c)  # img_prev 被删 -> 走标准路径; 换有 img_prev 但 state=None 见 R3b
    d = abs(lc.item() - lt.item())
    report('R3a 删除 img_prev 后 temporal 模型 ≡ 对照', d == 0.0, f'损失差 {d}')
    # R3b: img_prev 在但 state=None(forward_train_pair 注入关闭)
    head = mt.model[-1]
    with torch.no_grad():
        feats = mt._fpn_features(batch['img'])
        dec = head.forward_train_pair(feats, None, None)
        is_tensor = isinstance(dec[0], torch.Tensor)
    report('R3b state=None 注入关闭返回张量', is_tensor,
           f'dec_bboxes 类型={type(dec[0]).__name__}（M=0 应退化为张量）')

    # ---------- R4: 梯度流(state detach / adapter 有梯度 / backbone 有梯度)
    mt.zero_grad()
    head = mt.model[-1]
    mt.eval()
    with torch.no_grad():
        state = head.extract_pair_state(mt._fpn_features(batch['img_prev']))
    mt.train()
    no_grad_ok = all(not t.requires_grad for t in state['hidden'].values())
    preds = head.forward_train_pair(mt._fpn_features(batch['img']), None, state)
    dec_b = [preds[2], *preds[0]] if isinstance(preds[0], list) else None
    dec_s = [preds[3], *preds[1]] if isinstance(preds[1], list) else None
    bs = batch['img'].shape[0]
    bidx = batch['batch_idx'].to(DEVICE, dtype=torch.long).view(-1)
    targets = {'cls': batch['cls'].to(DEVICE, dtype=torch.long).view(-1),
               'bboxes': batch['bboxes'].to(DEVICE),
               'batch_idx': bidx,
               'gt_groups': [(bidx == i).sum().item() for i in range(bs)]}
    loss = mt.criterion((dec_b, dec_s), targets)
    total = sum(loss.values())
    total.backward()
    g_adapter = head.temporal_adapter.weight.grad
    g_embed = head.temporal_src_embed.grad
    g_bb = mt.model[1].conv.weight.grad if hasattr(mt.model[1], 'conv') else None
    report('R4 梯度流', no_grad_ok and g_adapter is not None and g_adapter.abs().sum() > 0
           and g_embed is not None,
           f'state 无梯度={no_grad_ok}, adapter 梯度范数={g_adapter.abs().sum():.3e}, '
           f'src_embed 范数={g_embed.abs().sum():.3e}')

    # ---------- R5: 注入后各层宽度
    shapes = [tuple(t.shape[1:2]) for t in preds[0]] if isinstance(preds[0], list) else []
    report('R5 T33 层宽 [300,300,325]', shapes == [(300,), (300,), (325,)],
           f'各层 query 数={shapes}')

    # ---------- R6: 恒等 adapter 下 T33 注入 ≡ QD 探针 eval T33(解码器级逐位)
    mt.zero_grad()
    mt.eval()
    with torch.no_grad():
        feats_t = mt._fpn_features(batch['img'])
        feats_p = mt._fpn_features(batch['img_prev'])
        state = head.extract_pair_state(feats_p)
        feats_, shapes_ = head._get_encoder_input(feats_t)
        embed, refer, _, _ = head._get_decoder_input(feats_, shapes_)
        embed_add = head.temporal_adapter(state['hidden'][3]) + head.temporal_src_embed
        refer_add = state['boxes'][3].clamp(1e-4, 1 - 1e-4)
        head.decoder.train()  # 训练分支(逐层 list); decoder 内无 BN, 与 eval 分支数值一致
        out_tr = head.decoder(embed, refer, feats_, shapes_, head.dec_bbox_head,
                              head.dec_score_head, head.query_pos_head,
                              inject_at={2: (embed_add, refer_add)})
        head.decoder.eval()
        # QD 探针路径(整模型 eval, T33/m25)
        head.set_temporal_enabled(True)
        head.set_query_overrides(temporal_queries=25)
        head.set_temporal_config(source_depth=3, injection_layer=2)
        head.reset_temporal_state()
        _ = mt(batch['img_prev'])  # 写入 temporal state
        y_probe, x_probe = mt(batch['img'])
        head.set_temporal_enabled(False)
    dec_probe = x_probe[0][-1]  # (bs, 325, 4)
    dec_train = out_tr[0][-1]
    d6 = (dec_probe - dec_train).abs().max().item()
    report('R6 恒等注入 ≡ QD-T33', d6 == 0.0,
           f'末层框最大差 {d6}（应逐位一致）')

    # ---------- R7: extract_pair_state 无残留
    with torch.no_grad():
        s1 = head.extract_pair_state(feats_p)
        s2 = head.extract_pair_state(feats_p)
    same = all(torch.equal(a, b) for a, b in
               zip(s1['hidden'].values(), s2['hidden'].values()))
    report('R7 状态提取幂等', same, f'两次提取逐位一致={same}')

    # ---------- R8: checkpoint 保存/重载(b1a 属性与输出保持)
    import io
    buf = io.BytesIO()
    torch.save({'model': mt}, buf)
    buf.seek(0)
    ck = torch.load(buf, map_location='cpu', weights_only=False)
    mt2 = ck['model'].to(DEVICE).eval()
    ok_spec = mt2.model[-1].b1a_temporal_spec == head.b1a_temporal_spec
    ok_adp = hasattr(mt2.model[-1], 'temporal_adapter')
    with torch.no_grad():
        ya = mt(img)
        yb = mt2(img)
    d8 = (ya[0] - yb[0]).abs().max().item()
    report('R8 ckpt 往返', ok_spec and ok_adp and d8 == 0.0,
           f'spec 保持={ok_spec}, adapter 在={ok_adp}, 输出差 {d8}')

    # ---------- R9: DN 关闭
    dn_ok = preds[4] is None
    dn_zero = all(float(v) == 0.0 for k, v in loss.items() if k.endswith('_dn'))
    report('R9 DN 关闭', dn_ok and dn_zero,
           f'dn_meta is None={dn_ok}, dn 损失全零={dn_zero}')

    # ---------- R10: eval 注入路径应用 adapter(与手动一致)
    with torch.no_grad():
        head.temporal_adapter.weight.mul_(1.0)  # 保持恒等但走应用分支
        head._temporal_state = {
            'hidden': {3: state['hidden'][3]}, 'boxes': {3: state['boxes'][3]},
            'velocity': torch.zeros_like(state['boxes'][3]), 'idx': state['idx']}
        head.temporal_motion_alpha = 0.0
        head.temporal_source_depth = 3
        g = head._gather_temporal_state()
        manual = head.temporal_adapter(state['hidden'][3]) + head.temporal_src_embed
        d10 = (g[0] - manual).abs().max().item()
        head._temporal_state = None
    report('R10 eval 路径 adapter 应用', d10 == 0.0, f'与手动应用差 {d10}')

    # ---------- 汇总
    n_fail = sum(1 for _, ok, _ in results if not ok)
    LOGGER.info(f'\n回归结果: {len(results) - n_fail}/{len(results)} 通过')
    out = str(EVAL_DIR / 'regression')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'b1a_regression.json'), 'w', encoding='utf-8') as f:
        json.dump([{'id': r, 'pass': bool(ok), 'detail': d} for r, ok, d in results],
                  f, ensure_ascii=False, indent=2)
    if n_fail:
        LOGGER.warning(f'{n_fail} 项未过, 禁止开训')
        sys.exit(1)
    LOGGER.info('全部通过, 允许开训')


if __name__ == '__main__':
    main()
