
[TOC]

# Jetson-PI 部署与加速效果测试

面向本机（Jetson / Ubuntu）从零跑通环境，并测试 Jetson-PI（FAAC 异步推理）的实际效果。

本仓库主要验证两件事：

1. **延迟**：`serve_policy` 单次推理耗时（`policy_infer_ms` / `server_infer_ms`）
2. **异步加速可用性**：在模拟推理延迟 \(\Delta\) 下，FAAC 能否维持 LIBERO 成功率（对应论文表格）

> 板端 llama.cpp 加速引擎在独立仓库 [Jetson-PI-Edge](https://github.com/PKU-SEC-Lab/Jetson-PI-Edge)。本仓库先用 JAX policy server + LIBERO sim 验证算法侧加速收益。

---

## 0. 前置条件

| 项 | 说明 |
|----|------|
| 已完成 | `git submodule update --init --recursive`（`third_party/libero` 已就绪） |
| JetPack | 本仓库 `jax[cuda12]` 需要 **JetPack 6 / L4T R36（CUDA 12）**。本机若仍是 **JP5 / R35.4.1**，见 [§11](#11-已知问题jetpack-5--jaxcuda12)–[§12](#12-升级-jetpack-5--6解决-cuda-驱动不匹配) |
| Python | 训练/服务端需要 **3.11**；JP5 自带 3.8 不够用 |
| 磁盘 | 建议预留 **≥ 40 GB**（权重 + venv + 日志/视频）。根分区紧张时先清 Docker 等再下模型 |
| 显示/仿真 | 建议安装 `xvfb`；没有则脚本会退回 `MUJOCO_GL=egl` |
| GPU | `nvpmodel` 可切到更高功耗模式；正式测延迟前可跑 `sudo jetson_clocks` |

```bash
cd /home/termitech/stephen/01-code/Jetson-PI

# 建议：提高性能模式（按需）
sudo nvpmodel -m 0          # 具体模式号以 nvpmodel -q --verbose 为准
sudo jetson_clocks
```

GitHub 不稳时可先：

```bash
git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000
```

---

## 1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"   # 或把 ~/.local/bin 加入 PATH
uv --version
```

---

## 2. 安装训练 / Policy Server 环境（JAX）

```bash
cd /home/termitech/stephen/01-code/Jetson-PI
export PYTHONNOUSERSITE=1

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# π0.5 需要的 transformers 补丁
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

验证：

```bash
export PY="$(pwd)/.venv/bin/python"
export PY_SERVER="$PY"
"$PY_SERVER" -c "import jax; print('jax', jax.__version__); print(jax.devices())"
```

期望：打印 `jax 0.5.3`，且 `jax.devices()` 为 `CudaDevice`，**无** `cudaErrorInsufficientDriver` / cuDNN handle 报错。

> **本机实测（Jetson AGX Orin DevKit + L4T R35.4.1）：** 会出现驱动不够、GPU JAX 不可用。现象、原因与 **升级 JetPack** 步骤见 [§11](#11-已知问题jetpack-5--jaxcuda12) / [§12](#12-升级-jetpack-5--6解决-cuda-驱动不匹配)。

---

## 3. 安装 LIBERO 评估 Client

```bash
cd /home/termitech/stephen/01-code/Jetson-PI

uv venv --python 3.11 examples/libero/.venv
source examples/libero/.venv/bin/activate

uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy=unsafe-best-match

uv pip install -e packages/openpi-client
uv pip install -e third_party/libero

# 仿真依赖（推荐）
sudo apt-get update
sudo apt-get install -y xvfb libegl1-mesa-dev
```

---

## 4. 下载预训练权重（ModelScope）

权重：[zebinyang/Jetson-PI-pi05](https://www.modelscope.cn/models/zebinyang/Jetson-PI-pi05)

```bash
cd /home/termitech/stephen/01-code/Jetson-PI
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

## 5. 设置环境变量（每次新开终端）

```bash
cd /home/termitech/stephen/01-code/Jetson-PI

export PYTHONNOUSERSITE=1
export REPO="$(pwd)"
export PY="${REPO}/.venv/bin/python"
export PY_SERVER="${PY}"
export PI0_CHECKPOINT="${REPO}/checkpoints/jetson-pi-pi05/pi05_libero"
export WM="${REPO}/checkpoints/jetson-pi-pi05/future_correction_module"
export CUDA_VISIBLE_DEVICES=0
export PORT=8000

# 检查
test -d "${PI0_CHECKPOINT}/params" && echo "PI0 OK"
test -d "${WM}/params" && echo "WM OK"
```

---

## 6. 测延迟（加速效果第一步）

用 `simple_client` 压测 policy server，看单次推理耗时。

### 6.1 启动 server

```bash
cd /home/termitech/stephen/01-code/Jetson-PI
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PY_SERVER}" -u scripts/serve_policy.py --env LIBERO --port "${PORT}" \
  --world-model-checkpoint "${WM}" \
  --world-model-token-reducer-kind learned_cross_attn \
  --world-model-action-encoder-kind transformer_block \
  --async-ae-proprio-source prefix_t \
  policy:checkpoint --policy.config pi05_libero --policy.dir "${PI0_CHECKPOINT}"
```

看到日志里有 `server listening`，且 `curl http://127.0.0.1:${PORT}/healthz` 返回 200 后再测。

### 6.2 跑 timing client（另一终端）

```bash
cd /home/termitech/stephen/01-code/Jetson-PI
source .venv/bin/activate
export PYTHONNOUSERSITE=1

python examples/simple_client/main.py \
  --env libero \
  --host 127.0.0.1 \
  --port 8000 \
  --num-steps 50 \
  --timing-file logs/timing_libero.parquet
```

关注表中的：

- `policy_infer_ms`：模型推理
- `server_infer_ms` / `client_infer_ms`：服务端与端到端

同一硬件上可对比「仅 π0.5」与「π0.5 + WM」的耗时（去掉 `--world-model-*` 参数再起一次 server）。

---

## 7. LIBERO 异步评估（验证 FAAC 加速可用性）

异步参数约定（默认 \(H=10\)）：

| 符号 | 含义 | 脚本变量 |
|------|------|----------|
| \(H\) | action horizon | `AH`（默认 10） |
| \(K\) | 异步触发步 | `K`（默认 9） |
| \(\Delta\) / overlap | 执行与预测错位，\(\Delta=H-K\) | `OVERLAP` |

\(\Delta\) 越大，模拟的推理延迟越大；论文关心的是大 \(\Delta\) 下成功率是否仍高。

### 7.1 冒烟（先确认能跑通）

```bash
cd /home/termitech/stephen/01-code/Jetson-PI

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

### 7.2 正式单点（FAAC，对应论文 Ours）

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

### 7.3 置信度调度（对应论文 +Sched）

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

### 7.4 换任务套件

```bash
export LIBERO_WM_EVAL_TASK_SUITE=libero_object   # 或 libero_goal / libero_10
bash scripts/eval_wm_libero_spatial.sh
```

---

## 8. 如何解读「加速效果」

1. **延迟数字**：来自第 6 节 `simple_client` 的 `*_infer_ms`  
   - 异步推理的价值：用 chunk 执行掩盖这段延迟，控制频率不必等于推理频率。
2. **算法侧加速收益**：在第 7 节增大 \(\Delta\) 时看成功率  
   - 基线异步方法（VLASH/RTC）在大 \(\Delta\) 时 SR 掉得快  
   - FAAC（Ours / +Sched）应更接近同步水平（见 README 表格）
3. **对照论文**：同一模型扫 \(\Delta=1..9\)，对比 `libero_spatial/object/goal/10` 的 SR。

从 `client.log` 末尾汇总成功率；视频在对应 `logs/.../videos/`。

---

## 9. 可选：本仓库三阶段训练（非加速测试必需）

仅在需要自己训 future correction 时：

```bash
export OPENPI_LIBERO_LOCAL_DATASET_DIR=PATH/TO/DATASET/libero
export PI0_CHECKPOINT="${REPO}/checkpoints/jetson-pi-pi05/pi05_libero"
export PY="${REPO}/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
# 训练建议 ≥48GB 级显存；Jetson 共享内存可能不够，优先用下载好的 WM

bash scripts/train_wm_libero_spatial_four_stage.sh
```

---

## 10. 板端引擎（真正的 onboard 加速）

算法验证通过后，板端量化/加速推理走：

- 仓库：[PKU-SEC-Lab/Jetson-PI-Edge](https://github.com/PKU-SEC-Lab/Jetson-PI-Edge)
- 技术栈：llama.cpp（+ FlashRT 等）

本仓库的 JAX `serve_policy` 用于训练与 LIBERO 评估，不是最终 Orin 量产推理路径。

---

## 11. 已知问题：JetPack 5 + `jax[cuda12]`

### 11.1 现象（本机）

```text
平台: Jetson AGX Orin Developer Kit
系统: Ubuntu 20.04 + L4T R35.4.1（JetPack 5.1.x）
驱动: NVIDIA UNIX Open Kernel Module 35.4.1（CUDA 11.4 一代）

$ .venv/bin/python -c "import jax; print(jax.__version__); print(jax.devices())"
jax 0.5.3
E.... cuda_dnn.cc:502] ... cudaErrorInsufficientDriver :
  CUDA driver version is insufficient for CUDA runtime version
[CudaDevice(id=0)]   # 仅枚举到设备；cuDNN / 实际 GPU 计算不可用
```

有时后续再 import 会直接落到 `CpuDevice`。

### 11.2 原因

| 侧 | 版本 |
|----|------|
| 系统驱动 / L4T | **R35.4.1** → 用户态 CUDA **11.4** |
| `uv sync` 装的 `jax[cuda12]==0.5.3` | 自带 **CUDA 12.9** runtime + cuDNN 9.x wheel |

驱动太旧，无法加载 CUDA 12.9 runtime → `cudaErrorInsufficientDriver`。  
**这不是安装步骤漏了，而是 JP5 与本仓库默认 CUDA12 JAX wheel 不兼容。**

### 11.3 可选出路

| 方案 | 适用 |
|------|------|
| **A. 升级到 JetPack 6（L4T R36 + Ubuntu 22.04 + CUDA 12）** | 要在本机用 JAX GPU 跑 `serve_policy` / LIBERO |
| B. 换一台 x86 + 大显存、CUDA 12 机器跑本仓库评估 | 不想动板端系统 |
| C. 板端加速走 [Jetson-PI-Edge](https://github.com/PKU-SEC-Lab/Jetson-PI-Edge) | 测 llama.cpp 引擎，而非 JAX |
| D. 临时 CPU 冒烟（`CUDA_VISIBLE_DEVICES=""`） | 只验证 import / 脚本链路，**不能**测加速 |

推荐在本机继续测 JAX 延迟时走 **方案 A**，步骤见下一节。

---

## 12. 升级 JetPack 5 → 6（解决 CUDA 驱动不匹配）

> 官方说明：**不支持**用 `apt upgrade` 从 JetPack 5 升到 6。  
> 参考：[JetPack 6.0](https://developer.nvidia.com/embedded/jetpack-sdk-60)、[Image-Based OTA（L4T R36）](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/SoftwarePackagesAndTheUpdateMechanism.html)。

本机当前：`R35.4.1`。NVIDIA 要求：

```text
R35.4.1  →  先升到 R35.5.0  →  再升到 R36.x（JetPack 6）
```

并对 A/B 启动链：**同一目标版本的 OTA 通常要跑两遍**（两条 boot chain 都升上去）。

### 12.1 升级前检查与备份

```bash
cat /etc/nv_tegra_release          # 应类似 R35 ... REVISION: 4.1
cat /proc/device-tree/model        # 本机: Jetson AGX Orin Developer Kit
df -h /                            # 建议可用 ≥ 20G；OTA 包体积大
```

- **整盘备份 / 拷走** `~/stephen`、密钥、docker 数据等重要文件（升级或刷机可能丢数据）。
- 准备一台 **x86_64 Linux 主机**（生成 OTA payload 或跑 SDK Manager），与 Jetson 同网或 USB。
- 确认板型配置名（AGX Orin DevKit 为 `jetson-agx-orin-devkit`）。

### 12.2 路径选择

| 路径 | 特点 |
|------|------|
| **A. SDK Manager 刷机** | 步骤少；默认会重装 rootfs（**清空用户数据**）；适合可重装的开发机 |
| **B. Image-Based OTA** | 可在设备上更新分区；从 35.4.1 须 **两跳**（→35.5→36）；步骤多，需主机生成 payload |

---

### 12.3 路径 A：SDK Manager 刷 JetPack 6（推荐「可接受重装」时）

在 **Ubuntu x86 主机**上：

1. 安装 [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager)。
2. Jetson 关机，USB-C 连主机，进 **Force Recovery**：
   - AGX Orin DevKit：按住 **Recovery**，点 **Reset/Power**，松开 Recovery。
   - 主机执行 `lsusb`，应看到类似 `NVIDIA Corp. APX`。
3. 打开 SDK Manager → 选中检测到的 AGX Orin → 选择 **JetPack 6.x**（建议较新的稳定版，如 6.1/6.2，对应 L4T R36.x）。
4. 勾选 Jetson Linux / 需要的 CUDA、cuDNN 等 → Flash。
5. 按提示完成首次开机、用户创建；系统应为 **Ubuntu 22.04**。

刷完后在 Jetson 上验证：

```bash
cat /etc/nv_tegra_release          # 应为 R36 (release), REVISION: x.x
cat /etc/lsb-release               # Ubuntu 22.04
# 驱动应能匹配 CUDA 12
```

然后 **重建** 本仓库环境（旧 `.venv` 建议删掉重装）：

```bash
cd /home/termitech/stephen/01-code/Jetson-PI
rm -rf .venv
# 回到本文 §1–§2 重新 uv sync，再跑 JAX 验证
```

---

### 12.4 路径 B：Image-Based OTA（35.4.1 → 35.5.0 → 36.x）

以下在 **x86 主机**准备 BSP，在 **Jetson** 上执行 `nv_ota_start.sh`。包名/版本号以 [NVIDIA L4T 下载页](https://developer.nvidia.com/embedded/jetson-linux-archive) 为准；示例用 R35.5.0 与 R36.4.x/R36.5.x。

#### 步骤 1：主机下载并展开 R35.5.0 + R36.x

```bash
# 示例目录布局（主机）
mkdir -p ~/nvidia-ota/{r35.5,r36}
# 从 NVIDIA 下载（需登录）并解压到上述目录：
#   jetson_linux_r35.5.0_aarch64.tbz2
#   tegra_linux_sample-root-filesystem_r35.5.0_aarch64.tbz2
#   ota_tools_R35.5.0_aarch64.tbz2
#   jetson_linux_r36.<x>_aarch64.tbz2
#   tegra_linux_sample-root-filesystem_r36.<x>_aarch64.tbz2
#   ota_tools_R36.<x>_aarch64.tbz2
```

对每个 BSP：

```bash
cd ~/nvidia-ota/r35.5
tar xf jetson_linux_r35.5.0_aarch64.tbz2
sudo tar xpf tegra_linux_sample-root-filesystem_r35.5.0_aarch64.tbz2 -C Linux_for_Tegra/rootfs
cd Linux_for_Tegra && sudo ./apply_binaries.sh

# 对 r36 目录做同样操作（版本号替换为你下载的 R36.x）
```

把对应 `ota_tools_*.tbz2` 解压进各自的 `Linux_for_Tegra`（与官方 OTA 文档一致）。

#### 步骤 2：生成「35.4 → 35.5」payload，并在板子上刷第一跳

官方要求从 **早于 35.5.0** 的版本不能直接 OTA 到 R36，须先到 35.5.0。  
在主机按 [Image_based_OTA_Examples](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/SoftwarePackagesAndTheUpdateMechanism.html) 生成：

```bash
# 主机：BASE=当前板子版本对应的 BSP 树，TARGET=R35.5.0
# AGX Orin DevKit:
cd ${TARGET_BSP}   # R35.5.0 的 Linux_for_Tegra
sudo BASE_BSP=${BASE_BSP} ./tools/ota_tools/version_upgrade/l4t_generate_ota_package.sh \
  jetson-agx-orin-devkit R35-4
# 产出 bootloader/.../ota_payload_package.tar.gz
```

拷到 Jetson：

```bash
# 在 Jetson 上
sudo mkdir -p /ota
# 将 ota_payload_package.tar.gz 与 ota_tools_R35.5.0_aarch64.tbz2 放到可访问路径
cd ~
tar xjpf ota_tools_R35.5.0_aarch64.tbz2
cd Linux_for_Tegra/tools/ota_tools/version_upgrade
sudo ./nv_ota_start.sh /ota/ota_payload_package.tar.gz
sudo reboot
```

重启后确认：

```bash
cat /etc/nv_tegra_release   # REVISION 应为 5.0
```

**对 A/B 链再执行一次同样 OTA**（官方：两条 chain 都要升到 35.5.0）。

#### 步骤 3：生成「35.5 → 36.x」payload，第二跳

```bash
# 主机
cd ${R36_TARGET_BSP}   # R36.x 的 Linux_for_Tegra
sudo BASE_BSP=${R35_5_BSP} ./tools/ota_tools/version_upgrade/l4t_generate_ota_package.sh \
  jetson-agx-orin-devkit R35-5
```

再拷到 Jetson，用 **R36 的** `ota_tools` 执行 `nv_ota_start.sh`，reboot；并对第二条 boot chain **再 OTA 一次**。

```bash
cat /etc/nv_tegra_release   # 应为 R36
cat /etc/lsb-release        # 一般为 Ubuntu 22.04
```

#### 步骤 4：升级后恢复本仓库环境

```bash
cd /home/termitech/stephen/01-code/Jetson-PI
rm -rf .venv examples/libero/.venv
# 按 §1–§3 重装；再验证 GPU JAX：
.venv/bin/python -c "import jax; print(jax.__version__, jax.devices()); import jax.numpy as jnp; print((jnp.ones((2,2))@jnp.ones((2,2))).block_until_ready())"
```

无 `InsufficientDriver`，且 matmul 在 GPU 上成功，即可继续 §4 之后的加速测试。

### 12.5 升级后仍失败时

- 确认 `cat /proc/driver/nvidia/version` 已是 R36 一代驱动。
- 确认没用错旧容器 / 旧 `LD_LIBRARY_PATH` 指向 JP5 库。
- 删除并重建 `.venv`，避免混用旧 CUDA 12.9 wheel 与残缺系统库。
- 若只关心板端引擎延迟，可改走 §10 Jetson-PI-Edge，不依赖 JAX GPU。

---

## 13. 其他常见问题

| 现象 | 处理 |
|------|------|
| `cudaErrorInsufficientDriver` / cuDNN handle 失败 | 见 [§11](#11-已知问题jetpack-5--jaxcuda12)、[§12](#12-升级-jetpack-5--6解决-cuda-驱动不匹配) |
| `No module named 'jax'` | `PY_SERVER` 必须指向 `.venv/bin/python` |
| OOM | 降 `LIBERO_WM_EVAL_NUM_TRIALS`；`export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7`；`XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| 缺 `norm_stats` | `PI0_CHECKPOINT` 指到含 `assets/.../norm_stats.json` 的 `pi05_libero` |
| LIBERO / EGL 报错 | 装 `xvfb`，或 `MUJOCO_GL=egl` / `glx` |
| GitHub TLS / GnuTLS -110 | `git config --global http.version HTTP/1.1` 后重试 |
| 磁盘不足 | 先清 Docker（曾占 ~26G）；权重与日志勿堆满根分区 |
| port 占用 | 换 `PORT`，或 `fuser -k ${PORT}/tcp` |

---

## 14. 推荐执行顺序（本机 checklist）

- [x] submodule（已完成）
- [x] 安装 `uv` + JAX server venv + transformers patch
- [ ] **升级 JetPack 5.1（R35.4.1）→ JetPack 6（R36）**（否则 GPU JAX 不可用，见 §11–§12）
- [ ] 升级后重建 `.venv`，确认 `jax.devices()` 为 CUDA 且无 InsufficientDriver
- [ ] 安装 LIBERO client venv + `xvfb`
- [ ] ModelScope 下载 `Jetson-PI-pi05`
- [ ] `simple_client` 测延迟，记录 `policy_infer_ms`
- [ ] `LIBERO_WM_EVAL_NUM_TRIALS=2` 冒烟
- [ ] `NUM_TRIALS=50`，扫多个 \(\Delta\)（改 `K`）得到 SR 曲线
- [ ] （可选）开 `LIBERO_WM_EVAL_ADAPTIVE_KAPPA=1` 对比 +Sched
- [ ] （后续）接入 Jetson-PI-Edge 测板端引擎加速
