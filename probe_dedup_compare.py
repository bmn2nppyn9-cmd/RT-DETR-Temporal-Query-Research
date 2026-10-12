# -*- coding: utf-8 -*-
"""
probe_dedup_compare.py —— 去重后的公平对比
==========================================
问题: 录制预测经 postprocess 展平(query,class)取 top-300, 同一框会以不同类别
      重复出现(实测重复率 39%), 导致 copy/extrap 只有 ~183 个不同位置,
      而 encoder 的 300 个提案位置各不相同 —— 数量上不公平。
做法: 对 copy/extrap 先去重再算 recall, 并额外报"取前 N 个"的控制变量,
      看 encoder 的优势是来自"位置更好"还是仅仅"位置更多"。
"""
import sys, json, os
import numpy as np
sys.path.insert(0,'.')
from eval_sap import iou_matrix, load_gt

DATA=r'D:\lunwen\自研\数据集'
def log(m): print(m,flush=True)

with open(os.path.join(DATA,'argoverse-hd','stream_index.json'),encoding='utf-8') as f:
    idx=json.load(f)
gt=load_gt(os.path.join(DATA,'argoverse-hd','annotations_val.json'))
z=np.load('stream_preds/preds_d6.npz',allow_pickle=False)
rec={k:z[k] for k in z.files}
per={}
for i in range(len(rec['scene'])):
    s,e=int(rec['ptr'][i]),int(rec['ptr'][i+1])
    per[(int(rec['scene'][i]),int(rec['frame_idx'][i]))]=(rec['box'][s:e],rec['score'][s:e])

targets=[]
for sc in idx['scenes']:
    sid=sc['scene_id']
    for fr in sc['frames']:
        if fr['idx']>=2 and (sid,fr['idx']) in per:
            targets.append((sid,fr['idx'],fr['file']))
log(f'评测帧 {len(targets)}')

def rec_at(d,g,thr):
    if len(g)==0: return 0,0
    if len(d)==0: return 0,len(g)
    return int((iou_matrix(d,g).max(axis=0)>=thr).sum()), len(g)

def dedup(b,s):
    """按框去重, 保留同类最高分的那个"""
    if len(b)==0: return b,s
    o=np.argsort(-s,kind='stable'); b,s=b[o],s[o]
    _,keep=np.unique(np.round(b,1),axis=0,return_index=True)
    keep=np.sort(keep)
    return b[keep],s[keep]

stats={}
for name in ('copy','copy_dedup','copy_top300_from183','extrap','extrap_dedup'):
    stats[name]={0.3:[0,0],0.5:[0,0],0.7:[0,0]}
nb={k:0 for k in stats}

for sid,fi,fn in targets:
    b1,s1=per[(sid,fi-1)]
    b2,s2=per[(sid,fi-2)]
    bd,sd=dedup(b1,s1)
    # 运动矢量(用去重后的框)
    bdv,_=dedup(b1,s1)
    b2d,s2d=dedup(b2,s2)
    v=np.zeros_like(bdv)
    if len(bdv) and len(b2d):
        iou=iou_matrix(bdv,b2d); best=iou.argmax(axis=1)
        ok=iou[np.arange(len(bdv)),best]>=0.3
        v[ok]=bdv[ok]-b2d[best[ok]]
    ex=bdv+0.5*v
    probes={
        'copy':b1,
        'copy_dedup':bd,
        'copy_top300_from183':bd,
        'extrap':b1+0.5*np.zeros_like(b1),
        'extrap_dedup':ex,
    }
    gb,_=gt[fn]
    for k,d in probes.items():
        nb[k]+=len(d)
        for thr in (0.3,0.5,0.7):
            h,t=rec_at(d,gb,thr); stats[k][thr][0]+=h; stats[k][thr][1]+=t

log('')
log('='*78)
log(f"{'探针':<24}{'框数/帧':>10}{'R@0.3':>10}{'R@0.5':>10}{'R@0.7':>10}")
for k in stats:
    row=f"{k:<24}{nb[k]/max(len(targets),1):>10.1f}"
    for thr in (0.3,0.5,0.7):
        h,t=stats[k][thr]; row+=f"{100.0*h/max(t,1):>9.1f}%"
    log(row)
log('='*78)
json.dump({k:{str(thr):100.0*v[0]/max(v[1],1) for thr,v in s.items()} for k,s in stats.items()},
          open('out_probe_dedup.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
log('已写入 out_probe_dedup.json')
