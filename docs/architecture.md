# Architecture and failure semantics

## 控制层与执行层

`tqinfra` 的配置、编排、manifest 和 report 只依赖标准库。模型框架仅在隔离 worker 中加载。这使招聘方或没有 GPU 的开发者也能验证基础设施部分；模型 DLL 失败不会阻止历史数据面板。

配置以 config 文件所在目录为基准解析。worker 接收 TQ_DATA_ROOT、TQ_BACKEND、TQ_INITIAL_WEIGHTS、TQ_WORKSPACE，研究主链不再依赖用户 TONY 的目录。模型源码使用固定基准补丁恢复。

## 独占和产物保护

workspace/.worker.lock 使用排他创建，同一个 workspace 一次只允许一个 worker。前置检查在锁内完成，避免两个运行同时通过检查。不同 workspace 可并行，但这个锁不是 GPU 全局资源调度器：共享 GPU 的 benchmark 应单独运行。

prepare-pairs/record/evaluate 不覆盖已有 stage 输出；train 不覆盖已存在配置 checkpoint。regression 输出可重复生成。崩溃残留锁需要检查其 PID/host，确认 worker 已停止再移走；不自动删除可能属于活跃 worker 的锁。

## 状态和追溯

创建 manifest(running) → Popen(worker) → succeeded/failed/timed_out/cancelled。stdout/stderr 汇入 worker.log。Windows timeout 使用 taskkill /T，Unix 使用独立 session + killpg。进程启动失败也会落盘 failed。manifest 使用同目录临时文件 + replace。

记录 command、已解析配置、Python/OS、Git commit/dirty、主链代码 SHA256、输入 SHA256 和各 stage 产物 SHA256。Git 不可用时 provenance 字段为空但进程仍可执行。配置中的 backend 提交/API 可由 doctor --probe 检查；实验 manifest 不是模型/数据全集完整性证明。

哈希较大 checkpoint/NPZ 会产生一次额外 I/O；本地研究工作负载优先保证追溯，不将 manifest 写入时间算进模型 inference latency。

## 有状态推理

record worker 先关闭 temporal 预热 predictor，随后启用传播，在每个场景首帧 reset state。每条序列按时间顺序推理。不同来源保留 query/source/depth 元信息，便于拆源分析。录制的 predict wall time 不启用 decoder event 插桩，避免把额外同步混进端到端对比。

## 范围

这是单机 ML 实验平台，不包含在线 HTTP 服务、多租户身份体系、集群队列或高可用存储。研究训练属于外部数据/模型工作负载；CPU demo 和 CI 检验控制层可靠性，模型测试另行记录。历史探索脚本不是受管 API。
