# Validation record — 2026-10-12

本次实际执行结果，原始摘要见 [local-validation.json](verification/local-validation.json)。历史研究精度/性能与本次验证分开。

| 检查 | 结果 | 范围 |
|---|---|---|
| editable 安装 | 成功 | Windows，Python 3.13 |
| 控制层/指标测试 | 12 项通过 | 并发锁、退出状态、超时子进程清理、启动失败、产物、损坏证据、类别匹配、无输出 |
| Ruff | 通过 | tqinfra / tests / scripts |
| CPU demo | 成功 | fixture worker、manifest、SHA256、离线报告 |
| 浏览器页面 | 已用 Edge headless 截图核查 | 表格、预算控制、来源说明正常；截图见 assets/dashboard.png |
| fresh backend setup | 成功 | 从 GitHub 新 clone，固定提交，应用补丁；四文件与研究源码一致 |
| 真实数据 prepare-pairs | 成功 | 56 个场景、7876 对 |
| 真实模型回归 | 11/11 通过 | PyTorch 2.5.1 CPU；报告见 model-regression.json |
| 真实训练 smoke | 成功 | 基线/传播各 1 epoch，2 个帧对、2 张验证图，保存 best/last checkpoint |
| 刚训练权重推理 | 成功 | 两场景各三帧；传播 query 数 [300,325,325,300,325,325]，场景状态重置 |
| 刚训练权重评测 | 成功 | 全指标及拆源 JSON，非全量指标；JSON 禁止 NaN |

本次修复了无 GT 类别混入 AP/AR 均值、无检测工作点 argmax 以及 NaN 输出问题；全量历史八类都有 GT 的聚合公式不受无 GT 类别排除影响。归档历史 JSON 保留原值，没有将 smoke 数字补进论文结果。

默认 Anaconda 的 CUDA PyTorch 加载被 Windows 应用控制策略拦截。模型功能验证改用独立环境继承可用的 Python 3.11 / CPU PyTorch，未绕过系统策略。Polars 2.0.0 和 1.44.2 在本机出现 SSE3 检查错误，换成 1.32.3 后训练与 checkpoint 保存通过。

新验证不证明 GPU 加速、全量训练收敛、官方 COCO/sAP 对齐或线上 SLO。GitHub CI 另行执行 Windows/Linux 与 Python 版本矩阵；其结果应以 Actions 页面为准。

本机完整原始日志和产物位于 artifacts/ml-validation、artifacts/training-smoke-v3 和 artifacts/final-demo，未上传大型数据或权重。发布摘要记录测试当时的代码哈希和基础 Git 状态，较晚运行还记录 backend 哈希，便于区分测试与后续文档提交。

GitHub [首轮 CI](https://github.com/bmn2nppyn9-cmd/RT-DETR-Temporal-Query-Research/actions/runs/38245830899) 四组环境全部通过。

历史 b1ac25 录制被强制覆盖到 300 queries 的问题已修复；原 checkpoint 的新 CPU 六帧录制全部为 325 queries。这里只验证修复后的行为，不把该 smoke 当作全量精度重测。
