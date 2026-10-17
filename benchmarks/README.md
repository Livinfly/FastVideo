# FastVideo 可移植性能测试

本目录独立于 `exp/` 和其他框架。当前支持 FastH3 V1 / V2 的 T2AV 配置，
使用原生 `VideoGenerator`、VSA-H3 和完整模型权重。启动器支持多个命名配置；
每个模型的每个卡数组单独加载、预热、测量并保存结果。

## 环境与启动

在 **`exp-fixed-version` 分支**执行；实际启动前脚本会校验分支，发现不符即报错。
先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，再执行：

```bash
bash benchmarks/setup.sh
.venv/bin/python benchmarks/run.py \
  --model-path fasth3_v1=/models/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree \
  --model-path fasth3_v2=/models/FastVideo-FastH3-8-Step-V2 \
  --gpus 8 4 2 1 --validate-video \
  --output outputs/benchmarks/fasth3 --background
.venv/bin/python benchmarks/monitor.py outputs/benchmarks/fasth3 --watch
.venv/bin/python benchmarks/report.py outputs/benchmarks/fasth3
```

模型路径可以放在任意位置，需是完整 snapshot，包含 transformer、文本编码器、tokenizer、
video/audio VAE、两个 scheduler 及模型索引。下载对应的官方仓库：

- `FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`
- `FastVideo/FastVideo-FastH3-8-Step-V2`

不要只下载 transformer；V2 的 `fastvideo_inference.json` 要一并保留。
脚本离线加载模型，启动前检查配置和 scheduler shift，加载后再次检查有效 DMD 阶梯与 VSA-H3 后端。
权重不复制进结果目录；报告保存本次路径、配置 hash 和权重文件大小。

`setup.sh` 使用 uv、Python 3.12 和 `requirements.lock` 中已验证的 CUDA 12.6 / torch 2.12 环境，
从本仓库构建 Python + Triton 版 `fastvideo-kernel`，不按 GPU 名称选择实现。
所用硬件仍需支持该 PyTorch/Triton/BF16 运行环境。
`UV`、`BENCH_VENV`、`UV_CACHE_DIR`、`PYTHON_VERSION` 可以覆盖；`--check` 仅检查现有环境并保存 freeze。
运行器使用调用它的 Python 解释器；监控和报告仅依赖 Python 标准库，要求 Python 3.10+，建议使用同一个虚拟环境。

## 模型与推理配置

默认选中 `configs/fasth3_v1.json` 和 `configs/fasth3_v2.json`。
只运行某个模型时传一个 `--config`；实验配置可复制到任意目录：

```bash
.venv/bin/python benchmarks/run.py \
  --config benchmarks/configs/fasth3_v2.json \
  --model-path fasth3_v2=/models/FastVideo-FastH3-8-Step-V2 \
  --gpus 4 2 1 --prompt 'A quiet café in the morning.' --dry-run
```

`--config` 后可传多个文件。每个文件的 `name` 是独立模型/配方标识，
通过重复的 `--model-path NAME=PATH` 指定对应权重；不把机器路径写入共享配置。
要同时比较同一模型的两套策略，复制配置并改为两个不同的 `name` 即可。
`sampling` 调整分辨率、帧数、seed 等，`engine.offload` / `engine.compile` 控制原生执行策略，
`pipeline.experimental` 控制 VSA 等模型参数。`--prompt` / `--prompt-file` 可覆盖所选配置的 prompt。

当前适配器只支持 MiniMax-H3/FastH3 的 T2AV；增加别的模型家族需新增适配器和验证，
不能只替换模型路径就认为已支持。V1/V2 的区别如下：

| 配置 | DiT forwards | API steps（sigma 点数） | DMD 阶梯 | video/audio shift | VSA sparsity |
|---|---:|---:|---|---|---:|
| fasth3_v1 | 4 | 5 | 999, 749, 500, 250 | 12 / 3 | 0.9 |
| fasth3_v2 | 8 | 9 | 999, 874, 749, 624, 500, 375, 250, 125 | 10 / 3 | 0.8 |

共同 workload：1344×768、124 帧、24 FPS、CFG=1、tile=64，SP=卡数、TP=1。
卡数需整除模型 attention heads（当前为 56）。

沿用最新原生默认策略：DiT/文本编码器/VAE offload、融合与缓存等采用该分支 API 默认值，
不是前三个单模型框架的“全部 offload 关闭”配方。唯一保留的兼容修正是
`pin_cpu_memory=false`，历史测试曾出现 FSDP pinning 的 CUDA invalid argument；
可在配置中显式改回。使用 tile64 的 Triton 路由，清理继承的 `FASTVIDEO_*` 开关，
应用配置文件中的 `environment`。不自动重试、切换 offload 或降低工作量。
如果原生设备策略开启 `lazy_module_load`，当前计时适配器会明确报错，避免 hooks 提前加载或延长模型生命周期；
这类设备需显式设置 `engine.offload.lazy_module_load=false` 并确认显存容量。

## 计时与报告

默认 **dit 模式（E2E + DiT）、每模式 1 次完整 warmup + 1 次正式测量**。
`--modes dit e2e --repeats 3` 开启双模式对照；每模式独立 warmup，正式请求按轮交替模式顺序。

- [`worker.py`](worker.py) 标注 **E2E START/END**：原生 `generate` 调用至返回，
  包含文本/噪声、DiT、scheduler、通信、双 VAE、视频/音频编码封装与写盘。
- [`timing.py`](timing.py) 标注 **DiT START/END**：完整 transformer forward 的 root hooks，
  开始 hook 置于已有 pre-hooks 前，结束 hook 置于已有 post-hooks 后，覆盖 SP/FSDP。
  CUDA events 在计时外预分配、初始化；无逐步同步。每个 forward 包含模型全部层。
- `e2e` 请求完全移除 DiT hooks，不记录 events；纯 e2e suite 也不分配 events。
  两模式均不加 pipeline 返回后的 CUDA 同步，不保留额外中间 tensor；
  原生 `FASTVIDEO_STAGE_LOGGING` 保持关闭，避免引入分阶段 CUDA 同步。
- `response_e2e_s` 为 rank 0 保存返回的时长；`distributed_e2e_s` 额外计至保存后 barrier 返回，
  每个 rank 的原始值均保存，再取最大。非 rank 0 的 response E2E 为 null。
- DiT 主指标 `max(sum)`：每 rank 先累计全部 forward，再取最大；另保存 `sum(max)` 和逐 forward events。
  scheduler、VAE 不属于 DiT。offload/通信在 hook 内发生的部分属于 DiT，不能解释为纯 kernel 时间。
- 加载、等待、准备、warmup、指标聚合及输出验证排除。音频有限值和输出形状在计时后校验；
  `--validate-video` 额外完整解码检查视频尺寸/帧数/FPS、音轨和文件 hash。

每组完成后更新 `report.md` / `summary.json`，包括配置、环境、E2E、DiT、
**同一模型且同一计时模式的单卡加速比**、warmup 对照、可选双模式逐轮差异。
单次标为“单次”，标准差 null；多次 CV > 2% 或至少三次连续变快且首尾下降 > 2% 时提示复查。
没有可用的已完成单卡基准时，加速比留空。

历史对比默认关闭。按需用 `--baseline /previous/suite/summary.json` 比较同格式结果，
启动时和独立报告命令均支持；不带参数重新生成即去掉历史章节与字段。
旧 `exp` 测试曾含额外同步、阶段计时日志和不同策略，不能直接当作同口径结果。

## 等待、状态与文件

GPU 被占用时默认等待：存在计算进程、已用显存 >256 MiB 或利用率非零，均视为忙。
每 30 秒检查，连续两次空闲才启动；进入推理后不再轮询。
`CUDA_VISIBLE_DEVICES` 支持 nvidia-smi 索引或唯一 UUID 前缀，每组选择其中前 N 张。
选卡与加锁使用完整 UUID；子进程使用原生 FastVideo 支持的数字索引及 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。
每个 rank 在加载模型前核对 CUDA 实际 UUID；若 CUDA/NVML 编号顺序不一致则报错停止，避免使用未加锁的 GPU。
此兼容处理全部位于 benchmark 脚本，不修改或运行时替换框架代码，也不在计时期间查询设备。
与 Hunyuan、LongCat、Kandinsky 共用同机同用户 `/tmp/video-benchmark-gpus-UID/` 文件锁，
锁持有至该组进程退出；外部不遵守锁的程序仍可占用 GPU。此实现面向整卡，不支持 MIG。
`--wait-for /another/suite/status.json` 额外等待已有实验完成；前置实验失败或进程消失时报错。

`status.json` 自动原子更新 `status`、`current`、`completed`、`waiting_reason`、
`has_failures`、`failure_count`、`last_error`、`launcher_error` 和更新时间。
错误摘要含阶段、退出码、日志及 rank 错误文件位置。加载/预热明确 OOM 记录后继续下一组；
正式测量 OOM 或其他错误停止矩阵。失败提示不会被后续成功组覆盖。
状态文件用于查看，手动修改不会触发重试或恢复。启动前校验失败尚未创建 suite；
进程被强制杀死后不能自行写状态，监控会检测进程是否仍存在。

```text
suite/
  plan.json, status.json, report.md, summary.json, model_manifest.json
  scripts/                       # 本次运行使用的完整计时代码/配置快照
  fasth3_v1/gN/                  # fasth3_v2 同样结构
    inference.log, command.sh, progress.json, effective_config.json
    metrics.json                 # 包含每 rank 原始时长
    error_rankN.json             # 失败时生成
    *.mp4, cache/rankN/
```

已有 suite 拒绝覆盖。运行期间不要修改推理源码、模型和 suite 快照。
CPU 检查：`.venv/bin/python -m unittest discover -s benchmarks -p 'test_*.py' -v`。
