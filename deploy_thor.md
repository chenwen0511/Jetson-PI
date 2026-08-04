# Jetson-PI 部署与加速效果测试（AGX Thor）

面向本机 **Jetson AGX Thor Developer Kit** 从零跑通环境，并测试 Jetson-PI（FAAC 异步推理）的算法侧加速效果。

本仓库主要验证两件事：

1. **延迟**：`serve_policy` 单次推理耗时（`policy_infer_ms` / `server_infer_ms`）
2. **异步加速可用性**：在模拟推理延迟 \(\Delta\) 下，FAAC 能否维持 LIBERO 成功率（对应论文表格）

> Orin 机型请看 [deploy_agx.md](./deploy_agx.md)。  
> 板端 llama.cpp 加速引擎在独立仓库 [Jetson-PI-Edge](https://github.com/PKU-SEC-Lab/Jetson-PI-Edge)。本仓库先用 JAX policy server + LIBERO sim 验证算法侧加速收益。

---

## 0. 本机现状（已核对）

| 项 | 本机实测 | 说明 |
|----|----------|------|
| 板型 | `NVIDIA Jetson AGX Thor Developer Kit` | Blackwell，compute capability **11.0（sm_110）** |
| L4T / JetPack | **R39.2.0**（JetPack **7.2**） | **无需**再升 JP6（那是 Orin JP5 的问题） |
| OS | Ubuntu **24.04.4** | |
| 驱动 / CUDA | Driver **595.78**，CUDA Toolkit **13.2** | 系统已装 `libcudnn9-cuda-13` |
| 内存 / 磁盘 | ~**122 GiB** RAM，根分区 ~**576G** 可用 | 远超权重+venv 建议的 40G |
| 功耗模式 | 当前已是 **MAXN（mode 0）** | 可选：`120W` / `90W` / `70W` |
| 系统 Python | **3.12**（无预装 3.11） | 用 `uv` 拉 **3.11** 即可 |
| 仓库路径 | `/home/nvidia/stephen/01-code/Jetson-PI` | |

与 Orin 文档的关键差异：

| | Orin（`deploy_agx.md`） | Thor（本文） |
|--|-------------------------|--------------|
| 默认 JAX | `jax[cuda12]==0.5.3` | **不能**直接用：无 sm_110 / CUDA13 支持 |
| 驱动问题 | JP5 会 `InsufficientDriver` | 驱动已够新；瓶颈是 **JAX 版本与 GPU arch** |
| 官方 Jetson JAX | JP6 → ~0.6.x | L4T ≥38 → **`jax 0.10.x`（Blackwell）** |

硬件层面：**完全支持**做本仓库加速验证，且比 Orin 更合适。软件上必须按下面步骤换掉仓库默认的 CUDA12 / jax 0.5.3。

---

## 1. 前置条件

| 项 | 说明 |
|----|------|
| submodule | `git submodule update --init --recursive`（确保 `third_party/libero` 完整） |
| Python | 训练/服务端用 **3.11**（`requires-python >=3.11`；勿直接用系统 3.12 硬扛整包） |
| 磁盘 | 建议预留 **≥ 40 GB**（权重 + venv + 日志/视频） |
| 显示/仿真 | 建议装 `xvfb`；没有则脚本会退回 `MUJOCO_GL=egl` |
| GPU 占用 | 正式测延迟前用 `nvidia-smi` 看是否有其他进程占 GPU；本机曾有 `run_inference_persistent.py` 占数 GB～二十多 GB |
| 性能 | MAXN 下可再锁频：`sudo jetson_clocks` |

```bash
cd /home/nvidia/stephen/01-code/Jetson-PI

# 确认板型与软件栈
cat /proc/device-tree/model
cat /etc/nv_tegra_release          # 期望 R39 ... REVISION: 2.0
nvidia-smi                         # NVIDIA Thor / CUDA 13.x

# 性能模式（本机通常已是 MAXN）
sudo nvpmodel -m 0                 # 0=MAXN；也可用 1=120W / 2=90W / 3=70W
nvpmodel -q
sudo jetson_clocks

# submodule
git submodule update --init --recursive
```

GitHub 不稳时可先：

```bash
git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000
```

---

## 2. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"   # 或把 ~/.local/bin 加入 PATH
uv --version
```

---

## 3. 安装训练 / Policy Server 环境（JAX on Thor）

仓库 `pyproject.toml` 锁定的是 `jax[cuda12]==0.5.3`，在 Thor 上会因 **sm_110 / CUDA 13** 失败或无法真正跑 GPU（社区实测无法为 0.5.3 编出 Thor kernel；NVIDIA / jetson-containers 对 L4T≥38 默认构建 **jax 0.10.0**）。

推荐流程：**先装项目依赖并排除默认 JAX，再装 Thor 可用的 `jax[cuda13]`**。

### 3.1 创建 venv 并同步依赖（覆盖 JAX）

```bash
cd /home/nvidia/stephen/01-code/Jetson-PI
export PYTHONNOUSERSITE=1

# 用 3.11（uv 会自动下载）
uv venv --python 3.11 .venv
source .venv/bin/activate

# 同步除 jax 系列外的依赖，避免拉下 cuda12 / 0.5.3
GIT_LFS_SKIP_SMUDGE=1 uv sync --no-install-package jax \
  --no-install-package jaxlib \
  --no-install-package jax-cuda12-plugin \
  --no-install-package jax-cuda12-pjrt

GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 安装支持 CUDA 13 / Blackwell 的 JAX（版本以当时 PyPI 为准；优先 ≥0.10）
uv pip install -U "jax[cuda13]"

# π0.5 需要的 transformers 补丁
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

若 `uv sync --no-install-package ...` 仍把 cuda12 plugin 装进来，可强制卸掉再装 cuda13：

```bash
uv pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt \
  jax-cuda13-plugin jax-cuda13-pjrt 2>/dev/null || true
uv pip install -U "jax[cuda13]"
```

### 3.2 备选：jetson-containers / NGC 预编译 wheel

若 PyPI 的 `jax[cuda13]` 在 Thor 上出现 `no kernel image` / cusolver FFI 等错误，改用 NVIDIA 生态为 Thor 打的包：

- [jetson-containers `jax`](https://github.com/dusty-nv/jetson-containers/tree/master/packages/ml/jax)（L4T≥38 默认 **0.10.0**，含 sm_110）
- [NGC JAX](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax)（≥26.04 标注 Thor support）

把对应 `jaxlib` / `jax_cuda13_plugin` / `jax_cuda13_pjrt` wheel 装进 `.venv` 后，再 `uv pip install -e .`（同样不要让 cuda12 的 0.5.3 覆盖）。

参考讨论：[How to install jax==0.5.3 on Jetson Thor](https://forums.developer.nvidia.com/t/how-to-install-jax-0-5-3-on-jetson-thor-device/362519)（结论：**不要死磕 0.5.3**）。

### 3.3 验证 GPU JAX

```bash
export PY="$(pwd)/.venv/bin/python"
export PY_SERVER="$PY"

"$PY_SERVER" -c "
import jax, jax.numpy as jnp
print('jax', jax.__version__)
print('devices', jax.devices())
print('backend', jax.default_backend())
x = jnp.ones((1024, 1024))
y = (x @ x).block_until_ready()
print('matmul ok', float(y[0,0]))
"
```

期望：

- `jax.devices()` 为 `[CudaDevice(id=0)]`
- `default_backend` 为 `gpu`
- matmul **无** `no kernel image` / `InsufficientDriver` / cuDNN handle 报错
- 日志里出现 `Using hardcoded values for Thor` 一类提示可忽略

若 `flax` / `orbax-checkpoint` 与新 JAX 不兼容，按报错升级同栈版本后再测（本仓库原 pin：`flax==0.10.2`、`orbax-checkpoint==0.11.13`）。能 `import openpi` 并完成上面 matmul 即可继续。

---

## 4. 安装 LIBERO 评估 Client

Client 用独立 venv；Thor 上优先装 **CUDA 13 / aarch64** 可用的 PyTorch，不要假设 x86 的 `cu121` wheel 一定合适。

```bash
cd /home/nvidia/stephen/01-code/Jetson-PI

uv venv --python 3.11 examples/libero/.venv
source examples/libero/.venv/bin/activate
export PYTHONNOUSERSITE=1

# 先装仿真与依赖（跳过过旧的 torch pin，再单独装）
uv pip install -r examples/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  --index-strategy=unsafe-best-match \
  || true

# 若上一步因 torch/aarch64 失败：去掉 torch 相关后重装其余包，再装 Jetson Torch
# 推荐：使用 NVIDIA 为 Jetson / Thor 提供的 PyTorch wheel，或 jetson-containers 的 torch 包
# 验证：
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

uv pip install -e packages/openpi-client
uv pip install -e third_party/libero

sudo apt-get update
sudo apt-get install -y xvfb libegl1-mesa-dev
```

> `examples/libero/requirements.txt` 里仍可能写着很老的 `torch==1.11.0+cu113`。在 Thor 上应以「能 `import torch` 且 `cuda.is_available()`」为准，不必死守该 pin。

---

## 5. 下载预训练权重（ModelScope）

权重：[zebinyang/Jetson-PI-pi05](https://www.modelscope.cn/models/zebinyang/Jetson-PI-pi05)

```bash
cd /home/nvidia/stephen/01-code/Jetson-PI
"$PY_SERVER" -m pip install modelscope

"$PY_SERVER" -c "
from modelscope import snapshot_download
snapshot_download('zebinyang/Jetson-PI-pi05', local_dir='./checkpoints/jetson-pi-pi05')
"
```

目录结构应类似：

```text
checkpoints/jetson-pi-pi05/
  pi05_libero/                 # π0.5-LIBERO（含 params/、norm_stats）
  future_correction_module/    # FAAC future correction（含 params/）
```

**不要合并**两个目录的 `params/`。

---

## 6. 设置环境变量（每次新开终端）

```bash
cd /home/nvidia/stephen/01-code/Jetson-PI

export PYTHONNOUSERSITE=1
export REPO="$(pwd)"
export PY="${REPO}/.venv/bin/python"
export PY_SERVER="${PY}"
export PI0_CHECKPOINT="${REPO}/checkpoints/jetson-pi-pi05/pi05_libero"
export WM="${REPO}/checkpoints/jetson-pi-pi05/future_correction_module"
export CUDA_VISIBLE_DEVICES=0
export PORT=8000

# Thor 上建议关闭预分配，避免与其他 GPU 进程抢空闲显存
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

# 检查
test -d "${PI0_CHECKPOINT}/params" && echo "PI0 OK"
test -d "${WM}/params" && echo "WM OK"
nvidia-smi
```

---

## 7. 测延迟（加速效果第一步）

用 `simple_client` 压测 policy server，看单次推理耗时。

### 7.1 启动 server

```bash
cd /home/nvidia/stephen/01-code/Jetson-PI
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PY_SERVER}" -u scripts/serve_policy.py --env LIBERO --port "${PORT}" \
  --world-model-checkpoint "${WM}" \
  --world-model-token-reducer-kind learned_cross_attn \
  --world-model-action-encoder-kind transformer_block \
  --async-ae-proprio-source prefix_t \
  policy:checkpoint --policy.config pi05_libero --policy.dir "${PI0_CHECKPOINT}"
```

看到日志里有 `server listening`，且 `curl http://127.0.0.1:${PORT}/healthz` 返回 200 后再测。

### 7.2 跑 timing client（另一终端）

```bash
cd /home/nvidia/stephen/01-code/Jetson-PI
source .venv/bin/activate
export PYTHONNOUSERSITE=1

mkdir -p logs
python examples/simple_client/main.py \
  --env libero \
  --host 127.0.0.1 \
  --port 8000 \
  --num-steps 50 \
  --timing-file logs/timing_libero_thor.parquet
```

关注表中的：

- `policy_infer_ms`：模型推理
- `server_infer_ms` / `client_infer_ms`：服务端与端到端

同一硬件上可对比「仅 π0.5」与「π0.5 + WM」的耗时（去掉 `--world-model-*` 参数再起一次 server）。  
Thor 上数字会明显好于 Orin，记录时注明板型与 JAX 版本，勿与 Orin 表格直接混比。

---

## 8. LIBERO 异步评估（验证 FAAC 加速可用性）

异步参数约定（默认 \(H=10\)）：

| 符号 | 含义 | 脚本变量 |
|------|------|----------|
| \(H\) | action horizon | `AH`（默认 10） |
| \(K\) | 异步触发步 | `K`（默认 9） |
| \(\Delta\) / overlap | 执行与预测错位，\(\Delta=H-K\) | `OVERLAP` |

\(\Delta\) 越大，模拟的推理延迟越大；论文关心的是大 \(\Delta\) 下成功率是否仍高。

### 8.1 冒烟（先确认能跑通）

```bash
cd /home/nvidia/stephen/01-code/Jetson-PI

export LIBERO_WM_EVAL_NUM_TRIALS=2          # 冒烟用；正式对比改回 50
export LIBERO_WM_EVAL_TASK_SUITE=libero_spatial
export AH=10
export K=9                                   # Δ=1
export OVERLAP=$((AH - K))
export PORT=8000

bash scripts/eval_wm_libero_spatial.sh
```

输出目录：`logs/<suite>_*_h${AH}_k${K}_*/`

- `client.log`：成功率、任务进度
- `serve.log`：server 日志
- `run_meta.txt`：本次参数
- `videos/`：回放

### 8.2 正式单点（FAAC，对应论文 Ours）

```bash
export LIBERO_WM_EVAL_NUM_TRIALS=50
export LIBERO_WM_EVAL_TASK_SUITE=libero_spatial
export AH=10
export K=9          # 改 K 即可扫不同 Δ；例如 K=5 → Δ=5
export OVERLAP=$((AH - K))
export PORT=8000

bash scripts/eval_wm_libero_spatial.sh
```

扫多个 \(\Delta\) 示例：

```bash
export LIBERO_WM_EVAL_NUM_TRIALS=50
export AH=10
for K in 9 7 5 3 1; do
  export K
  export OVERLAP=$((AH - K))
  export PORT=$((8000 + K))
  echo "==== H=${AH} K=${K} Δ=${OVERLAP} ===="
  bash scripts/eval_wm_libero_spatial.sh
done
```

### 8.3 置信度调度（对应论文 +Sched）

```bash
export LIBERO_WM_EVAL_ADAPTIVE_KAPPA=1
export LIBERO_WM_EVAL_KAPPA_DELTA=0.4
export LIBERO_WM_EVAL_NUM_TRIALS=50
export AH=10
export K=9
export OVERLAP=$((AH - K))

bash scripts/eval_wm_libero_spatial.sh
```

或直接跑 K 从 9→1 的 sweep：

```bash
export WM="${REPO}/checkpoints/jetson-pi-pi05/future_correction_module"
export PI0_CHECKPOINT="${REPO}/checkpoints/jetson-pi-pi05/pi05_libero"
export PY_SERVER="${REPO}/.venv/bin/python"
export LIBERO_WM_EVAL_KAPPA_DELTA=0.4
export LIBERO_WM_EVAL_NUM_TRIALS=50
export CUDA_VISIBLE_DEVICES=0

bash scripts/libero_wm_eval_spatial_k9to1_adaptive_kappa_low_replan_gpu2_kd0p4.sh
```

### 8.4 换任务套件

```bash
export LIBERO_WM_EVAL_TASK_SUITE=libero_object   # 或 libero_goal / libero_10
bash scripts/eval_wm_libero_spatial.sh
```

---

## 9. 如何解读「加速效果」

1. **延迟数字**：来自第 7 节 `simple_client` 的 `*_infer_ms`  
   - 异步推理的价值：用 chunk 执行掩盖这段延迟，控制频率不必等于推理频率。
2. **算法侧加速收益**：在第 8 节增大 \(\Delta\) 时看成功率  
   - 基线异步方法（VLASH/RTC）在大 \(\Delta\) 时 SR 掉得快  
   - FAAC（Ours / +Sched）应更接近同步水平（见 README 表格）
3. **对照论文**：同一模型扫 \(\Delta=1..9\)，对比 `libero_spatial/object/goal/10` 的 SR。  
   - 论文表格里的 Orin 延迟比例与 Thor 实测延迟不同；**SR–\(\Delta\) 曲线**仍可对照，**绝对 ms** 请单独记录为 Thor 结果。

从 `client.log` 末尾汇总成功率；视频在对应 `logs/.../videos/`。

---

## 10. 可选：本仓库三阶段训练（非加速测试必需）

Thor 统一内存更大，比 Orin 更有机会本机训 future correction；仍建议优先用已下载的 WM 做加速验证。

```bash
export OPENPI_LIBERO_LOCAL_DATASET_DIR=PATH/TO/DATASET/libero
export PI0_CHECKPOINT="${REPO}/checkpoints/jetson-pi-pi05/pi05_libero"
export PY="${REPO}/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

bash scripts/train_wm_libero_spatial_four_stage.sh
```

---

## 11. 板端引擎（真正的 onboard 加速）

算法验证通过后，板端量化/加速推理走：

- 仓库：[PKU-SEC-Lab/Jetson-PI-Edge](https://github.com/PKU-SEC-Lab/Jetson-PI-Edge)
- 技术栈：llama.cpp（+ FlashRT 等）

本仓库的 JAX `serve_policy` 用于训练与 LIBERO 评估。Edge 仓库是否已官方支持 Thor / sm_110，需单独对照该仓库说明；**不要默认 Orin 二进制可直接跑**。

---

## 12. Thor 上 JAX 常见问题

### 12.1 为什么不能直接 `uv sync`？

| 侧 | 版本 |
|----|------|
| 本机 | JetPack 7.2 / L4T R39 / CUDA **13.2** / GPU **sm_110** |
| 仓库默认 | `jax[cuda12]==0.5.3`（CUDA 12.x wheel，无 Blackwell 支持） |

结果通常是：装得上但 kernel 不可用、编译失败、或落到错误 backend。这不是漏步骤，而是 **默认依赖面向 Orin/CUDA12**。

### 12.2 推荐出路

| 方案 | 适用 |
|------|------|
| **A. `jax[cuda13]` ≥0.10（本文 §3）** | 本机直接跑 `serve_policy` / LIBERO（优先尝试） |
| B. jetson-containers / NGC 预编译 JAX | PyPI wheel 有 `no kernel image` 等错误时 |
| C. 换 x86 + 大显存 CUDA 机器跑评估 | 只关心算法 SR，不关心 Thor 延迟数字 |
| D. 板端走 Jetson-PI-Edge | 测 llama.cpp 引擎，而非 JAX |

### 12.3 验证仍失败时

- 确认 `uv pip show jax jaxlib` 是 **cuda13** 插件，而不是残留的 `jax-cuda12-*` / `0.5.3`。
- 确认 `nvidia-smi` 能看到 Thor，且无其他进程占满内存。
- 删除重建 `.venv`，避免混用 cuda12 / cuda13 plugin。
- 查看是否误设 `CUDA_VISIBLE_DEVICES=""` 或 `JAX_PLATFORMS=cpu`。

---

## 13. 其他常见问题

| 现象 | 处理 |
|------|------|
| `no kernel image is available for execution on the device` | JAX 未含 sm_110；换 §3 的 cuda13 / 0.10+ 或 jetson-containers 包 |
| `cudaErrorInsufficientDriver` | 在 Thor JP7.2 上少见；检查是否用了错误容器/旧 `LD_LIBRARY_PATH` |
| `No module named 'jax'` | `PY_SERVER` 必须指向 `.venv/bin/python` |
| OOM / 与其他进程抢 GPU | `nvidia-smi` 停掉无关推理；`XLA_PYTHON_CLIENT_PREALLOCATE=false`；降低 `MEM_FRACTION` |
| 缺 `norm_stats` | `PI0_CHECKPOINT` 指到含 `assets/.../norm_stats.json` 的 `pi05_libero` |
| LIBERO / EGL 报错 | 装 `xvfb`，或 `MUJOCO_GL=egl` / `glx` |
| Client 装不上旧 torch pin | 见 §4，改用 Jetson/Thor 可用的 PyTorch |
| GitHub TLS / GnuTLS -110 | `git config --global http.version HTTP/1.1` 后重试 |
| port 占用 | 换 `PORT`，或 `fuser -k ${PORT}/tcp` |
| `third_party/libero` 不完整 | `git submodule update --init --recursive` |

---

## 14. 推荐执行顺序（Thor checklist）

- [ ] `git submodule update --init --recursive`
- [ ] 确认 `R39` + CUDA 13 + `nvidia-smi` 可见 Thor；`nvpmodel` 为 MAXN（或目标功耗档）
- [ ] 安装 `uv`；用 Python 3.11 建 `.venv`
- [ ] **按 §3 安装 `jax[cuda13]`（不要用仓库默认 0.5.3/cuda12）**，matmul 验证通过
- [ ] 打 transformers 补丁；`uv pip install -e .` 可 import
- [ ] 安装 LIBERO client venv + 可用 Torch + `xvfb`
- [ ] ModelScope 下载 `Jetson-PI-pi05`
- [ ] `simple_client` 测延迟，记录 `policy_infer_ms`（标注 Thor + JAX 版本）
- [ ] `LIBERO_WM_EVAL_NUM_TRIALS=2` 冒烟
- [ ] `NUM_TRIALS=50`，扫多个 \(\Delta\)（改 `K`）得到 SR 曲线
- [ ] （可选）开 `LIBERO_WM_EVAL_ADAPTIVE_KAPPA=1` 对比 +Sched
- [ ] （后续）评估 Jetson-PI-Edge 是否支持 Thor，再测板端引擎加速
