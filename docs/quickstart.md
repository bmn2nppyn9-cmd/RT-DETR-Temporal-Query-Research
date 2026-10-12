# Quickstart

## CPU 演示

Python 3.10+，在仓库根目录执行 `python -m pip install -e .`，然后 `python -m tqinfra demo`。在输出目录查看 report.html、summary.json、workspace/runs/<run-id>/manifest.json 和 worker.log。演示不依赖 PyTorch；历史研究面板有明确测量范围。

## 模型环境

建议独立虚拟环境，按自己设备安装 PyTorch，随后执行：

```bash
python scripts/setup_backend.py
python -m pip install -e vendor/ultralytics -c requirements-ml-constraints.txt
python -m pip install -e ".[evaluation]"
```

setup_backend 只在新目录 clone 固定上游，应用研究补丁；拒绝覆盖现有 checkout。模型依赖以固定上游 pyproject 为准。PyTorch/CUDA 配置不由控制层强制替换。

## 数据布局

配置 data_root 指向同时包含以下两个目录的父目录：

```text
argoverse-hd/
  annotations_val.json
  stream_index.json
  stream_index_train.json
  images/<file from stream index>
argoverse-hd-yolo/
  images/train/*.jpg
  labels/train/*.txt
  images/val/*.jpg
  labels/val/*.txt
  train_sub_val.txt
```

数据从 Argoverse-HD 的可用来源自行获取并按其许可使用。`build_yolo_dataset.py`、`build_stream_index.py`、`make_train_subset.py` 提供转换/索引工具；先看其 --help。不要把旧 train/val 文本清单的原机器路径直接复制到新机器。

initial_weights 必须是匹配八类 Argoverse-HD、ResNet18 结构的基线 checkpoint。历史短训从已训练基线微调；任意 COCO 权重不是这个精度结果的替代品。`b1a_pair/b1a_checkpoints_e3_sha256.json` 保存历史传播 checkpoint 的哈希。外部数据/权重不属于 CPU demo 的必要条件。

## 配置与执行

复制 configs/local.example.json 为 configs/machine.local.json。path 字段相对配置文件解析，支持绝对路径和空格。修改 data_root、initial_weights、backend、workspace；设置 configs（五种研究配置）、epochs、batch、device、workers、limit_scenes、limit_frames、timeout_seconds。limit_frames=0 是完整场景，非零仅供 smoke；不应把帧子集评测当完整指标。

```bash
python -m tqinfra doctor --config configs/machine.local.json --probe
python -m tqinfra run train --config configs/machine.local.json --dry-run
python -m tqinfra run prepare-pairs --config configs/machine.local.json
python -m tqinfra run regression --config configs/machine.local.json
python -m tqinfra run train --config configs/machine.local.json
python -m tqinfra run record --config configs/machine.local.json
python -m tqinfra run evaluate --config configs/machine.local.json
```

真实完整训练建议 configs 包含全部五种配置；两配置是低成本入口。evaluate 要求 b1a0 基线。timeout_seconds=0 表示没有自动截止，非零表示 worker 超时上限。不要把短超时用于长训练后期待自动恢复。

## 输出与恢复

pairs/、checkpoints/、records/、evaluation/ 写入配置 workspace。runs/<id>/ 每次运行都有日志和 manifest。失败返回非零；旧 stage 产物不会被默默覆盖。失败后保留日志，检查部分产物，使用新 workspace 重跑或明确移动旧输出，再运行下一阶段。没有宣称断点自动恢复。

doctor 返回 1 表示 ML 前置条件未就绪，返回 2 表示配置或检查异常；run 传播 worker 退出码，超时为 124，取消为 130。独占锁冲突不会启动 worker。

## 已验证环境与兼容约束

本次模型功能验证使用 Python 3.11 / PyTorch 2.5.1 CPU / Ultralytics 固定版本 / Polars 1.32.3。Windows 上 Polars 1.44.2 与 2.0.0 在本机报 unknown feature flag: sse3，因此提供经过验证的 Polars 约束。默认 Anaconda 的 CUDA PyTorch 被系统应用控制策略拒绝加载，本次没有绕过策略，也没有重新报告 GPU 数字。
