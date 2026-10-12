# RT-DETR Temporal Query Research

RT-DETR 静态三层解码器与时序查询传播研究代码整理包。

## 包含的内容

- 核心方法：前帧 query state 提取、缓存、指定 decoder 层注入、来源 adapter，以及两帧训练路径。
- `b1a_pair/`：两帧数据准备、五组配置短训、连续帧录制、评测与回归。
- `query_prop_b0/`：冻结传播的负向实验与拆源诊断；其评测代码也被主流程调用。
- `query_depth_probe/`：逐层收敛 oracle、注入深度对照、动态退出探索。
- `ablation_2d/`：深度 × query 数的静态精度/计时消融及绘图。
- `ablation_3d/`：深度 × query 数 × 分辨率消融脚本；随包现有结果仅包含计时。
- 根目录：数据转换、流式索引、公共指标、性能测量和早期探索脚本。
- `patches/`：针对固定 Ultralytics 上游提交的核心研究补丁。
- `upstream_snapshot/`：四个修改后的完整源码文件，供查看；不是完整可安装的 Ultralytics。

## 当前状态

这是按原文件结构复制的研究整理包，不是已完成跨机器验证的发行版。复制保留原始内容；原研究目录没有改动。

许多脚本仍包含本机绝对路径，运行前须修改。详见 `PACKAGING_NOTES.md` 与 `LOCAL_PATHS.json`。没有包含数据集、权重、原始预测 NPZ、完整训练输出或内部汇报文档。

## 恢复核心依赖

以下命令应在本包根目录执行；需要安装 Git，并按自己的 CUDA 环境选择 PyTorch。

```powershell
git clone https://github.com/ultralytics/ultralytics.git vendor/ultralytics
git -C vendor/ultralytics checkout 5a64e9e65b5cf66876c8bdce44345b6b279f5ee3
git -C vendor/ultralytics apply --check ../../patches/ultralytics_temporal_query.patch
git -C vendor/ultralytics apply ../../patches/ultralytics_temporal_query.patch
python -m pip install -e vendor/ultralytics
python -m pip install scipy pycocotools
```

必须同步替换实验脚本里的旧 `sys.path.insert` 路径，避免加载原机器的库。基准环境记录与依赖声明见 `THIRD_PARTY/ultralytics_pyproject.toml`；它不是已经锁定的复现环境。

## 研究流程

1. 准备 Argoverse-HD，生成 YOLO 标签和 train/val stream index。
2. 准备或训练 RT-DETR-ResNet18 初始权重；本包没有提供权重。
3. 生成基础训练子集，再运行 `b1a_pair/make_pair_subset.py` 生成帧对索引和清单。
4. 检查数据、模型、权重和输出路径，运行 `b1a_pair/train_b1a.py`。
5. 使用 `b1a_pair/record_b1a.py` 连续帧录制，再用 `eval_b1a.py` 评测。
6. 使用静态消融和 benchmark 脚本补充实验。具体参数参见各脚本说明和 argparse。

## 精选结果与解释范围

当前 `b1a_pair/out_full_metrics_b1a_e3.json` 的 AP50：三层基线 0.3876803，等 query 对照 0.3882275，S3→L3 传播 0.4045258。对应 AP：0.2285697、0.2279507、0.2421811。

这是已有 3 epoch 短训结果，采用自实现评测口径；不代表本次重新运行，也不代表官方 COCO 指标已经验证。训练结果 JSON 的 `val_a0_metrics` 是 temporal 关闭的健康检查，不能拿来替代连续帧主评测。

- 静态三层的计时收益不能代替带传播模型的端到端加速结论。
- sAP 的公平延迟、预热和调度协议仍需核查。
- QD-O 是 oracle 分析；QD-P 当前实现没有获得实际加速。
- B0 是冻结传播失败对照。
- `eval_b1a.py` 中某些 Go/No-Go 判据来自冻结探针；训练态的跨独立模型 TP 保持/损失判据需要复审，不能只以自动 Go/No-Go 标签为结论。

## 来源与许可证

模型实现基于 Ultralytics；其原始 AGPL-3.0 文本保留在 `THIRD_PARTY/ULTRALYTICS_LICENSE.txt`，修改后文件保留原版权/许可证头。本整理包尚未替项目作者确定完整发布元数据和统一许可证。

`SOURCE_MANIFEST.json` 记录复制文件的来源、大小和 SHA256。`PACKAGE_CHECK.json` 记录打包校验。
