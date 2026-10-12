# Measurement protocol

## 静态历史网格

ablation_2d/out_grid_2d.json：深度 × query 共 30 点，640 输入。RTX 5070 Ti Laptop、FP32、batch=1、fused、合成输入网络前向；每配置预热和多轮交错同步计时。AP50 优先 *_full，否则 *_sub，面板明确显示 full/subset-0.2。前向预算不是视频捕获、图像解码、传输与预测后处理端到端 SLO。

ablation_3d/timing_3d.json：27 个深度/query/分辨率计时点，仅 latency。不同分辨率没有配套精度，不能把二维精度横移过来制造 Pareto 前沿。

## 短训历史指标

b1a_pair/out_full_metrics_b1a_e3.json：五配置 3 epoch，独立训练，21 场景 15062 帧。当前 AP 使用仓库自实现；verify_ap_vs_pycoco.py 是对齐工具，不表示每份结果已经通过官方 evaluator 验证。训练期 val_a0_metrics temporal 关闭且是子集，不是最终流式结果。

S3→L3 AP50 比三层训练基线高约 1.68pp，比错位 T31 仅高约 0.07pp；不能据此宣称深度对齐是唯一有效机制。静态三层收益不能替代传播模型的端到端测试。sAP 调度是自实现的丢弃式模型，不能宣称官方 streaming benchmark。

## 新测量要求

独占 GPU 或明确记录共享；同样 checkpoint、图像、batch、dtype、融合、预热和输出预算；分清 model CUDA、decoder CUDA 和 predict wall。随机/轮转交错，报告 P50/P95/P99，保存所有配置和硬件信息。端到端录制先预热后 reset，场景不串状态。

本次控制层 demo fixture 是正确性演示，不是速度测试。新 CPU 模型 smoke 也不等于 GPU 性能、全量 AP 或训练收敛。验证记录必须把历史数字与新执行分开。

## 历史对照审核

本次核查原 `records_e3/preds_b1ac25.npz` 发现：该配置训练时使用 325 current queries，但历史录制代码将 encoder query 数覆盖为 300，因此历史表中的 b1ac25 不是完整的同预算 325-query 推理对照。历史数字保留并注明，不据此宣称已经排除增加 query 的贡献。当前受管录制已修复为使用 checkpoint 的 num_queries，后续正式对照应重录并重新评测。三层基线与 S3→L3 的历史差值仍是对应两配置的已有结果。
