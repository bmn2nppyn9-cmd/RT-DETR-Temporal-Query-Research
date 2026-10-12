"""Validate recorded evidence and render an offline, standalone budget explorer."""

import html
import json
import math
from pathlib import Path

from .runner import sha256, write_json


def number(value, label, minimum=0, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"Invalid {label}: {value!r}")
    return float(value)


def load_evidence(root):
    root = Path(root)
    paths = [
        root / "ablation_2d/out_grid_2d.json",
        root / "ablation_3d/timing_3d.json",
        root / "b1a_pair/out_full_metrics_b1a_e3.json",
    ]
    grid, timing, training = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    rows = []
    for key, r in grid["results"].items():
        depth, queries = map(int, key.split("_"))
        scope = "full" if "map50_full" in r else "subset-0.2"
        row = {"depth": depth, "queries": queries, "resolution": 640, "accuracy_scope": scope}
        for target, source in (("p50_ms", "p50_ms_f"), ("p95_ms", "p95_ms_f"), ("p99_ms", "p99_ms_f")):
            row[target] = number(r[source], key + "." + source, minimum=1e-9)
        row["AP50"] = number(r.get("map50_full", r.get("map50_sub")), key + ".AP50", maximum=1)
        if not row["p50_ms"] <= row["p95_ms"] <= row["p99_ms"]:
            raise ValueError(f"Quantiles out of order: {key}")
        rows.append(row)
    latency_rows = []
    for key, r in timing.items():
        p50, p95, p99 = [number(r[k], key + "." + k, minimum=1e-9) for k in ("median_ms", "p95_ms", "p99_ms")]
        if not p50 <= p95 <= p99:
            raise ValueError(f"Quantiles out of order: {key}")
        latency_rows.append(
            {
                "depth": int(number(r["depth"], key + ".depth", minimum=1)),
                "queries": int(number(r["queries"], key + ".queries", minimum=1)),
                "resolution": int(number(r["resolution"], key + ".resolution", minimum=1)),
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
            }
        )
    train_rows = []
    for key in ("b1a0", "b1ac25", "b1at31", "b1at23", "b1at33"):
        train_rows.append(
            {
                "config": key,
                **{
                    metric: number(training[key][metric], key + "." + metric, maximum=1)
                    for metric in ("AP", "AP50", "AP75")
                },
            }
        )
    return {
        "schema_version": 1,
        "static": rows,
        "latency_only": latency_rows,
        "training": train_rows,
        "sources": [{"path": p.relative_to(root).as_posix(), "sha256": sha256(p)} for p in paths],
    }


def render(root, output):
    evidence = load_evidence(root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Escaping '<' prevents a supplied JSON value from terminating the script element.
    payload = json.dumps(evidence, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    training = "".join(
        "<tr><td>"
        + html.escape(r["config"])
        + "</td>"
        + "".join(f"<td>{r[m] * 100:.2f}</td>" for m in ("AP", "AP50", "AP75"))
        + "</tr>"
        for r in evidence["training"]
    )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RT-DETR · Performance Evidence</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#111827;color:#e5e7eb}body{max-width:1120px;margin:40px auto;padding:0 24px}h1{font-size:32px}h2{margin-top:36px}p{line-height:1.8;color:#b9c5d7}.cards{display:flex;gap:16px;flex-wrap:wrap}.card{background:#1e293b;border:1px solid #334155;padding:20px;flex:1;border-radius:10px}.card strong{display:block;font-size:28px;color:#67e8f9}.controls{display:flex;gap:20px;align-items:center;flex-wrap:wrap;padding:18px 0}input,select{padding:8px;background:#1e293b;color:white;border:1px solid #64748b;border-radius:5px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #334155}th{color:#94a3b8}.ok{color:#6ee7b7}.no{color:#fca5a5}code{color:#67e8f9}.scroll{overflow:auto}footer{margin:40px 0;color:#94a3b8;font-size:12px}
</style><h1>RT-DETR · Performance Evidence</h1>
<p>本地 ML 实验控制平面的历史数据视图。筛选预算、核查尾延迟与精度来源；所有数字读取仓库 JSON，没有重新运行 GPU 实验。</p>
<div class="cards"><div class="card"><strong>30</strong>二维静态配置</div><div class="card"><strong>27</strong>三维延迟配置</div><div class="card"><strong>5</strong>两帧短训对照</div></div>
<h2>静态配置预算筛选 · 640 输入</h2><p>RTX 5070 Ti Laptop，FP32、fused、batch=1、合成输入纯前向。精度优先全量，否则为 20% 子集。只在相同协议中筛选；预算通过不等同于端到端 SLO。</p>
<div class="controls"><label>预算 ms <input id="budget" type="number" min="0" step="0.5" value="15"></label><label>计时口径 <select id="quantile"><option value="p50_ms">P50</option><option value="p95_ms">P95</option><option value="p99_ms">P99</option></select></label><label><input id="full" type="checkbox">只看全量精度</label><span id="count"></span></div>
<div class="scroll"><table><thead><tr><th>Depth / Queries</th><th>AP50 %</th><th>精度来源</th><th>P50 ms</th><th>P95 ms</th><th>P99 ms</th><th>预算</th></tr></thead><tbody id="static"></tbody></table></div>
<h2>三维网格 · 延迟证据</h2><p>这 27 点只有计时数据，本页不会为不同分辨率补造精度或与二维精度拼成 Pareto 前沿。</p><div class="scroll"><table><thead><tr><th>Depth / Queries / Resolution</th><th>P50 ms</th><th>P95 ms</th><th>P99 ms</th></tr></thead><tbody id="latency"></tbody></table></div>
<h2>两帧训练 · 3 epoch</h2><p>21 场景、15062 帧，历史自实现同帧 AP 口径。五组独立训练，S3→L3 对错位对照优势较小。历史 b1ac25 训练 325 queries、录制覆盖为 300，不构成完整的等 query 推理对照。此表没有公平配对的传播端到端延迟，不能用静态三层计时替代。</p>
<table><thead><tr><th>配置</th><th>AP %</th><th>AP50 %</th><th>AP75 %</th></tr></thead><tbody>TRAINING_ROWS</tbody></table>
<footer id="sources"></footer><script id="evidence" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('evidence').textContent);
function update(){const budget=Number(document.getElementById('budget').value),q=document.getElementById('quantile').value,full=document.getElementById('full').checked;
const rows=data.static.filter(r=>!full||r.accuracy_scope==='full').sort((a,b)=>b.AP50-a.AP50);
document.getElementById('static').innerHTML=rows.map(r=>`<tr><td>${r.depth} / ${r.queries}</td><td>${(r.AP50*100).toFixed(2)}</td><td>${r.accuracy_scope}</td><td>${r.p50_ms.toFixed(2)}</td><td>${r.p95_ms.toFixed(2)}</td><td>${r.p99_ms.toFixed(2)}</td><td class="${r[q]<=budget?'ok':'no'}">${r[q]<=budget?'通过':'超预算'}</td></tr>`).join('');
document.getElementById('count').textContent=`${rows.filter(r=>r[q]<=budget).length} / ${rows.length} 通过`}
['budget','quantile','full'].forEach(id=>document.getElementById(id).addEventListener('input',update));update();
document.getElementById('latency').innerHTML=data.latency_only.map(r=>`<tr><td>${r.depth} / ${r.queries} / ${r.resolution}</td><td>${r.p50_ms.toFixed(2)}</td><td>${r.p95_ms.toFixed(2)}</td><td>${r.p99_ms.toFixed(2)}</td></tr>`).join('');
const footer=document.getElementById('sources');data.sources.forEach(s=>{const p=document.createElement('p');p.textContent=s.path+' · SHA256 '+s.sha256;footer.appendChild(p)});
</script></html>""".replace("TRAINING_ROWS", training).replace("PAYLOAD", payload)
    output.write_text(page, encoding="utf-8")
    write_json(output.with_suffix(".json"), evidence)
    return output
