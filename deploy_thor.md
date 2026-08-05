# Jetson-PI 部署与加速效果测试（AGX Thor）

面向本机 **Jetson AGX Thor Developer Kit** 从零跑通环境，并测试 Jetson-PI（FAAC 异步推理）的算法侧加速效果。

本仓库主要验证两件事：

1. **延迟**：`serve_policy` 单次推理耗时（`policy_infer_ms` / `server_infer_ms`）
2. **异步加速可用性**：在模拟推理延迟 \Delta 下，FAAC 能否维持 LIBERO 成功率（对应论文表格）

> Orin 机型请看 [deploy_agx.md](./deploy_agx.md)。  
> 板端 llama.cpp 加速引擎在独立仓库 [Jetson-PI-Edge](https://github.com/PKU-SEC-Lab/Jetson-PI-Edge)。本仓库先用 JAX policy server + LIBERO sim 验证算法侧加速收益。

---

## 0. 本机现状（已核对）


| 项             | 本机实测                                     | 说明                                            |
| ------------- | ---------------------------------------- | --------------------------------------------- |
| 板型            | `NVIDIA Jetson AGX Thor Developer Kit`   | Blackwell，compute capability **11.0（sm_110）** |
| L4T / JetPack | **R39.2.0**（JetPack **7.2**）             | **无需**再升 JP6（那是 Orin JP5 的问题）                 |
| OS            | Ubuntu **24.04.4**                       |                                               |
| 驱动 / CUDA     | Driver **595.78**，CUDA Toolkit **13.2**  | 系统已装 `libcudnn9-cuda-13`                      |
| 内存 / 磁盘       | ~**122 GiB** RAM，根分区 ~**576G** 可用        | 远超权重+venv 建议的 40G                             |
| 功耗模式          | 当前已是 **MAXN（mode 0）**                    | 可选：`120W` / `90W` / `70W`                     |
| 系统 Python     | **3.12**（无预装 3.11）                       | 用 `uv` 拉 **3.11** 即可                          |
| 仓库路径          | `/home/nvidia/stephen/01-code/Jetson-PI` |                                               |


与 Orin 文档的关键差异：


|               | Orin（`deploy_agx.md`）      | Thor（本文）                              |
| ------------- | -------------------------- | ------------------------------------- |
| 默认 JAX        | `jax[cuda12]==0.5.3`       | **不能**直接用：无 sm_110 / CUDA13 支持        |
| 驱动问题          | JP5 会 `InsufficientDriver` | 驱动已够新；瓶颈是 **JAX 版本与 GPU arch**        |
| 官方 Jetson JAX | JP6 → ~0.6.x               | L4T ≥38 → `jax 0.10.x`**（Blackwell）** |


硬件层面：**完全支持**做本仓库加速验证，且比 Orin 更合适。软件上必须按下面步骤换掉仓库默认的 CUDA12 / jax 0.5.3。

---



## 1. 前置条件


| 项         | 说明                                                                                 |
| --------- | ---------------------------------------------------------------------------------- |
| submodule | `git submodule update --init --recursive`（确保 `third_party/libero` 完整）              |
| Python    | 训练/服务端用 **3.11**（`requires-python >=3.11`；勿直接用系统 3.12 硬扛整包）                        |
| 磁盘        | 建议预留 **≥ 40 GB**（权重 + venv + 日志/视频）                                                |
| 显示/仿真     | 建议装 `xvfb`；没有则脚本会退回 `MUJOCO_GL=egl`                                                |
| GPU 占用    | 正式测延迟前用 `nvidia-smi` 看是否有其他进程占 GPU；本机曾有 `run_inference_persistent.py` 占数 GB～二十多 GB |
| 性能        | MAXN 下可再锁频：`sudo jetson_clocks`                                                    |


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



### 1.1 pip / uv 镜像（本机已验证）

国内网络下建议**两套镜像一起配**：PyPI 管依赖包；`uv venv --python` 下的是独立 CPython，**不走清华/阿里 PyPI**。


| 场景                                            | 镜像                               | 说明                                               |
| --------------------------------------------- | -------------------------------- | ------------------------------------------------ |
| `uv pip` / `uv sync` / `pip install`          | **只用清华**（默认）                     | 普通 Python 包；**不要**同时挂阿里作第二 index（易在大包下载时抽中阿里并超时） |
| 清华挂了再换                                        | **阿里**                           | 整次命令改 `UV_INDEX_URL`，勿与清华并列                      |
| `uv venv --python 3.11` / `uv python install` | **南大** `python-build-standalone` | 本机实测比直连 GitHub **快很多**                           |


每次新开终端可先导出：

```bash
# ① 依赖包：只用清华（本机推荐；大同步时更稳）
export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export UV_HTTP_TIMEOUT=300
export UV_INDEX_STRATEGY="unsafe-best-match"

# 清华整体慢/失败时，再整段改成阿里（二选一，不要叠两个）
# export UV_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
# export UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple"
# export PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"

# ② Python 解释器：南大（uv 拉 cpython-*-aarch64 时必须设，否则会慢/卡住）
export UV_PYTHON_INSTALL_MIRROR="https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/"
```

也可写入用户级配置（一次设置，长期生效；**推荐只挂清华**）：

```bash
mkdir -p ~/.config/pip ~/.config/uv

cat > ~/.config/pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
timeout = 300
EOF

cat > ~/.config/uv/uv.toml <<'EOF'
# 本机验证：uv 安装 Python 3.11 明显快于直连 GitHub
python-install-mirror = "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/"

# 只配一个默认 index，避免 uv 去阿里拉小包超时导致整次 sync 失败
[[index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
EOF
```

建 venv 前确认镜像已生效（任选其一）：

```bash
# 临时
export UV_PYTHON_INSTALL_MIRROR="https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/"
uv venv --python 3.11 .venv

# 或依赖 ~/.config/uv/uv.toml 里的 python-install-mirror
uv venv --python 3.11 .venv
```

若 `uv sync` 报某个小包（如 `inquirerpy`）从阿里 `operation timed out`：

```bash
# 1) 改成只用清华后重试（已下载的 wheel 会复用缓存）
export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export UV_HTTP_TIMEOUT=300
# 编辑 ~/.config/uv/uv.toml：删掉 aliyun 那一段 [[index]]

# 2) 再跑原来的 sync / pip 命令即可
```

> 日志里若出现大量 `nvidia-*-cu12` / `cudnn-cu12`：说明仍在装仓库默认的 **CUDA 12 JAX**，与 Thor（CUDA 13）路径不符，应按 §3 用 `--no-install-package jax`* 再装 `jax[cuda13]`，不要只靠重试镜像。  
> PyTorch / NVIDIA 专用 wheel 仍可能需要额外 `--extra-index-url`（见 §4）。  
> ModelScope 权重下载走 modelscope.cn，与 pip / uv Python 镜像无关。

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

推荐流程：**先装项目依赖并排除默认 JAX，再装 Thor 可用的** `jax[cuda13]`。

### 3.1 创建 venv 并同步依赖（覆盖 JAX）

```bash
cd /home/nvidia/stephen/01-code/Jetson-PI
export PYTHONNOUSERSITE=1
# 若未写 ~/.config，请先按 §1.1 导出 UV_INDEX_URL / PIP_INDEX_URL
# 以及 UV_PYTHON_INSTALL_MIRROR（南大；本机验证可大幅加速拉 Python）

# 用 3.11（uv 会自动下载；务必先配好 python-install-mirror）
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

# 关键点：仓库 pyproject.toml 有 override `ml-dtypes==0.4.1`（给旧 jax 0.5.3），
# 与 jax≥0.10 冲突会报：JAX requires ml_dtypes version 0.5 or newer。
# 必须在「仓库目录外」升级，否则 uv 会继续钉死 0.4.1：
(
  cd /tmp
  uv pip install --python "${PWD}/../stephen/01-code/Jetson-PI/.venv/bin/python" -U 'ml-dtypes>=0.5' \
    || uv pip install --python /home/nvidia/stephen/01-code/Jetson-PI/.venv/bin/python -U 'ml-dtypes>=0.5'
)

# π0.5 需要的 transformers 补丁
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

更稳妥的 `ml-dtypes` 一行（推荐直接复制）：

```bash
cd /tmp && uv pip install --python /home/nvidia/stephen/01-code/Jetson-PI/.venv/bin/python -U 'ml-dtypes>=0.5'
```

若 `uv sync --no-install-package ...` 仍把 cuda12 plugin 装进来，可强制卸掉再装 cuda13：

```bash
uv pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt \
  jax-cuda13-plugin jax-cuda13-pjrt 2>/dev/null || true
uv pip install -U "jax[cuda13]"
cd /tmp && uv pip install --python /home/nvidia/stephen/01-code/Jetson-PI/.venv/bin/python -U 'ml-dtypes>=0.5'
```



### 3.2 备选：jetson-containers / NGC 预编译 wheel

若 PyPI 的 `jax[cuda13]` 在 Thor 上出现 `no kernel image` / cusolver FFI 等错误，改用 NVIDIA 生态为 Thor 打的包：

- [jetson-containers](https://github.com/dusty-nv/jetson-containers/tree/master/packages/ml/jax) `jax`（L4T≥38 默认 **0.10.0**，含 sm_110）
- [NGC JAX](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax)（≥26.04 标注 Thor support）

把对应 `jaxlib` / `jax_cuda13_plugin` / `jax_cuda13_pjrt` wheel 装进 `.venv` 后，再 `uv pip install -e .`（同样不要让 cuda12 的 0.5.3 覆盖）。

参考讨论：[How to install jax==0.5.3 on Jetson Thor](https://forums.developer.nvidia.com/t/how-to-install-jax-0-5-3-on-jetson-thor-device/362519)（结论：**不要死磕 0.5.3**）。

### 3.3 验证 GPU JAX（务必按顺序，防整机卡死）

> **本机实测警告：** `import jax` / `jax.devices()` 可以很快返回，但一执行  
> `jnp.ones(...)` / matmul 等真正上 GPU 的算子时，**可能把 Thor 整机卡死**（统一内存被 JAX 默认预分配吃光，或 XLA 编译挂死）。  
> **禁止**在未设下面环境变量的交互式 `python` 里直接跑 GPU 算子；若已卡住，另开 SSH/`Ctrl+Alt+F3` 杀进程或重启。

**Step A — 只枚举设备（相对安全，本机已通过）：**

```bash
cd /home/nvidia/stephen/01-code/Jetson-PI
source .venv/bin/activate
export PYTHONNOUSERSITE=1

# 强制关闭预分配；限制占用比例（Thor 统一内存，默认预分配极易拖死系统）
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15

python -c "
import jax
print('jax', jax.__version__)
print('devices', jax.devices())
print('backend', jax.default_backend())
"
```

期望：`jax 0.10.x`、`[CudaDevice(id=0)]`、`backend gpu`。到这里即可确认「认出 GPU」；**先不要继续 matmul**。

**Step B — 小算子（另开终端待命；超时就杀，勿干等）：**

终端 1 准备好杀进程：

```bash
# 一旦卡死，在另一终端执行：
pkill -f 'python.*jax' || true
# 或： pgrep -af python ; kill <pid>
```

终端 2（必须先 export 上面三个 `XLA_*`）：

```bash
# 用 timeout，避免无限挂死
timeout 60s python -c "
import jax.numpy as jnp
x = jnp.ones((64, 64))
y = (x @ x).block_until_ready()
print('matmul ok', float(y[0, 0]))
"
```

- **60s 内打印 `matmul ok`**：GPU 计算可用，可继续后面章节。  
- **超时 / 整机无响应**：PyPI 的 `jax[cuda13]==0.10.2` 在本机 Thor 上不可靠 → **立刻停用该路径**，改走 §3.2（jetson-containers / NGC 含 sm_110 的预编译包），不要反复重试 matmul。  
- 跑 Step B 前用 `nvidia-smi` 确认没有其它占 GPU 的 `python`（例如历史遗留的 `run_inference_persistent.py`）。

若 `flax` / `orbax-checkpoint` 与新 JAX 不兼容，按报错升级同栈版本后再测（本仓库原 pin：`flax==0.10.2`、`orbax-checkpoint==0.11.13`）。**以 Step B 成功为准**，不要只看 `CudaDevice` 枚举。

---



## 4. 安装 LIBERO 评估 Client

Client 用独立 venv。仓库锁定的 `examples/libero/requirements.txt` 面向旧环境，在 Thor 上会连续踩坑：

| 问题 | 原因 |
|------|------|
| `torch==1.11.0+cu113` 无解 | aarch64 / 现代索引没有该 wheel |
| `llvmlite==0.36.0` 编译失败 | **只支持 Python &lt;3.10**，与 3.11 冲突（本机已复现） |
| `third_party/libero` 为空 | submodule 未 init |

**不要**照搬 Orin 的 `uv pip sync ... cu121`。按下面做：

```bash
cd /home/nvidia/stephen/01-code/Jetson-PI
git submodule update --init --recursive
# 确认：ls third_party/libero | head

uv venv --python 3.11 examples/libero/.venv
source examples/libero/.venv/bin/activate
export PYTHONNOUSERSITE=1
# §1.1 清华源

# ① 过滤不兼容 pin，并强制 py3.11 可用的 numba/llvmlite/numpy
grep -vE '^(torch|torchvision|torchaudio|llvmlite|numba|numpy)==' \
  examples/libero/requirements.txt > /tmp/libero_req_filtered.txt

cat > /tmp/libero_overrides.txt <<'EOF'
numba>=0.59.0
llvmlite>=0.42.0
numpy>=1.23.5,<2.0.0
EOF

uv pip install "numpy>=1.23.5,<2" "llvmlite>=0.42" "numba>=0.59"
uv pip install -r /tmp/libero_req_filtered.txt --override /tmp/libero_overrides.txt

# ② PyTorch（二选一；评估 client 用 CPU 更稳）
# 方案 B（推荐先跑通）：CPU
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 方案 A（需要 client 侧 CUDA）：Jetson AI Lab SBSA / CUDA 13
# uv pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130
# 若缺 libcudss.so.0：sudo apt-get install -y libcudss0-cuda-13

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# ③ openpi-client + libero
# 注意：本机 `uv pip install -e third_party/libero` 会生成空 MAPPING 的 editable finder，
# `import libero` 失败。改用 --no-editable，或设 PYTHONPATH。
uv pip install -e packages/openpi-client
# libero 用 --no-deps 时装包本身；其运行时依赖（尤其 bddl）需另装，否则
# `from libero.libero.envs import OffScreenRenderEnv` → ModuleNotFoundError: bddl
uv pip install -r third_party/libero/requirements.txt \
  --override /tmp/libero_overrides.txt || true
# 若上面被旧 pin 卡住，至少保证评估链路：
uv pip install 'bddl==1.0.1' easydict future cloudpickle
uv pip install --no-deps --no-editable third_party/libero
# 若仍 import 失败：
# export PYTHONPATH="$(pwd)/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"

# ④ 预写 LIBERO config，跳过首次 import 的「Y/N: custom dataset path」交互
REPO="$(pwd)"
mkdir -p "${REPO}/.libero_eval" ~/.libero
cat > "${REPO}/.libero_eval/config.yaml" <<EOF
assets: ${REPO}/third_party/libero/libero/libero/assets
bddl_files: ${REPO}/third_party/libero/libero/libero/bddl_files
benchmark_root: ${REPO}/third_party/libero/libero/libero
datasets: ${REPO}/third_party/libero/libero/datasets
init_states: ${REPO}/third_party/libero/libero/libero/init_files
EOF
cp "${REPO}/.libero_eval/config.yaml" ~/.libero/config.yaml
export LIBERO_CONFIG_PATH="${REPO}/.libero_eval"

sudo apt-get update
sudo apt-get install -y xvfb libegl1-mesa-dev libglfw3 libglfw3-dev
```

冒烟（GLFW 装好后；无显示时用 EGL）：

```bash
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$(pwd)/.libero_eval"   # 避免停在 Y/N
python -c "import torch, mujoco, robosuite, openpi_client; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); from libero.libero import benchmark; print('libero ok')"
```

> 大模型在 JAX `serve_policy`；client 主要仿真 + websocket。优先 **CPU Torch**，减少再卡死风险。  
> 若仍卡在旧 pin：确认用的是 `/tmp/libero_req_filtered.txt` + `--override`，而不是原始 `requirements.txt`。

---



## 5. 下载预训练权重（ModelScope）

权重：[zebinyang/Jetson-PI-pi05](https://www.modelscope.cn/models/zebinyang/Jetson-PI-pi05)

```bash
cd /home/nvidia/stephen/01-code/Jetson-PI

# PY_SERVER 未设置时：用系统/conda python；已建 .venv 时：
#   export PY_SERVER="$(pwd)/.venv/bin/python"
: "${PY_SERVER:=$(command -v python3)}"

# modelscope 包走清华/阿里（§1.1）；权重本身从 modelscope.cn 拉
"$PY_SERVER" -m pip install -U modelscope \
  -i "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

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

# pip / uv 国内源（若已写 ~/.config 可省略）
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR:-https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/}"

# Thor 统一内存：务必关预分配；比例宁低勿高（本机曾因默认预分配/大 matmul 整机卡死）
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

# 检查
test -d "${PI0_CHECKPOINT}/params" && echo "PI0 OK"
test -d "${WM}/params" && echo "WM OK"
nvidia-smi
```

---



## 7. 测延迟（加速效果第一步）

用 `simple_client` 压测 policy server，看单次推理耗时。

> **本机注意**
> 1. `serve_policy` 必须用**仓库根** `.venv`（`PY_SERVER=${REPO}/.venv/bin/python`），不要用 `examples/libero/.venv`（那是仿真 client）。
> 2. 若报 `No module named 'tyro'`：根 `.venv` 还缺 openpi 依赖。不要盲目 `uv sync`（会拉 `lerobot`→`av` 源码编译失败，还可能把 JAX 降级）。按 §3.1 装完 JAX 后，再装最小服务端依赖（见下方 §7.0），并重新确认 `jax==0.10.x` + `jax_cuda13_plugin`。
> 3. 启动前看 `nvidia-smi`：若已有其它 `python` 占数 GB～二十多 GB，**先停掉**再开 server，否则易再次整机卡死。

### 7.0 补齐服务端依赖（仅当缺 tyro/flax 等）

```bash
cd /home/nvidia/stephen/01-code/Jetson-PI
source .venv/bin/activate   # 根 .venv
export PYTHONNOUSERSITE=1
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

uv pip install -e . --no-deps
uv pip install \
  'tyro>=0.9.5' 'flax>=0.12' 'orbax-checkpoint==0.11.13' \
  'transformers==4.53.2' 'einops>=0.8.0' 'equinox>=0.11.8' \
  'jaxtyping==0.2.36' 'beartype==0.19.0' 'rich>=14.0.0' \
  'polars>=1.30.0' 'pillow>=11.0.0' 'opencv-python>=4.10.0.84' \
  'tqdm-loggable>=0.2' 'typing-extensions>=4.12.2' 'filelock>=3.16.1' \
  'treescope>=0.1.7' 'sentencepiece>=0.2.0' 'numpydantic>=1.6.6' \
  'dm-tree>=0.1.8' 'augmax>=0.3.4' 'flatbuffers>=24.3.25' \
  'ml_collections==1.0.0' 'numpy>=1.22.4,<2.1' 'imageio>=2.36.1' \
  'wandb>=0.19.1' 'fsspec[gcs]>=2024.6.0' 'tensorstore==0.1.74' \
  pytest chex
# serve_policy 会 import torch（JAX 负责 GPU；此处用 CPU torch 更稳）
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e packages/openpi-client --no-deps
uv pip install websockets msgpack

# 防止被 override / sync 拉回旧版本
cd /tmp && uv pip install --python /home/nvidia/stephen/01-code/Jetson-PI/.venv/bin/python \
  -U 'jax[cuda13]==0.10.2' 'ml-dtypes>=0.5' 'numpy>=1.26,<2.1' \
  'orbax-checkpoint>=0.11.25' 'flax>=0.12'
# 说明：
# - orbax==0.11.13 与 jax 0.10 不兼容（DeviceLocalLayout）
# - flax==0.10.2 与 jax 0.10 不兼容（仍传 concrete= 给 jax.checkpoint）
#   需 flax>=0.12；清华源可能没有新 flax，用 --index-url https://pypi.org/simple

cp -r /home/nvidia/stephen/01-code/Jetson-PI/src/openpi/models_pytorch/transformers_replace/* \
  /home/nvidia/stephen/01-code/Jetson-PI/.venv/lib/python3.11/site-packages/transformers/

.venv/bin/python -c "import tyro, flax, jax; print(tyro.__version__, jax.__version__)"
```

### 7.1 启动 server

先清 GPU 占用并激活**根**环境：

```bash
nvidia-smi
# 若有无关 python 占显存，先停掉（确认不是你自己的任务）：
#   sudo kill <pid>
# 或： pgrep -af 'run_inference_persistent|python' 

deactivate 2>/dev/null || true
cd /home/nvidia/stephen/01-code/Jetson-PI
source .venv/bin/activate          # 必须是仓库根 .venv，不是 examples/libero/.venv
# 重新 export §6 全部变量（含 XLA_* / PY_SERVER / PI0_CHECKPOINT / WM / PORT）
```

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
  --env LIBERO \
  --host 127.0.0.1 \
  --port 8000 \
  --num-steps 50 \
  --timing-file logs/timing_libero_thor.parquet
```

关注表中的：

- `policy_infer_ms`：仓库计时字段（**JAX 路径下常未 `block_until_ready`，会严重偏低**，见 §7.3）
- `server_infer_ms` / `client_infer_ms`：服务端整次 `infer` 墙钟与端到端（当前更可信）

同一硬件上可对比「仅 π0.5」与「π0.5 + WM」的耗时（去掉 `--world-model-*` 参数再起一次 server）。  
Thor 上数字会明显好于 Orin，记录时注明板型与 JAX 版本，勿与 Orin 表格直接混比。

### 7.3 本机实测结果（AGX Thor，已跑通）

> **日期：** 2026-08-05  
> **板型：** Jetson AGX Thor Developer Kit（L4T R39.2 / JetPack 7.2，CUDA 13.2）  
> **栈：** `jax[cuda13]==0.10.2` + `flax>=0.12` + `orbax-checkpoint>=0.12`；`serve_policy` = π0.5-LIBERO + FAAC WM  
> **压测：** `simple_client`，`--env LIBERO`，`num-steps=50`  
> **产物：** `logs/timing_libero_thor.parquet`

| Metric | Mean (ms) | Std | P50 | P95 | P99 |
|--------|-----------|-----|-----|-----|-----|
| `policy_infer_ms` | **6.7** | 0.2 | 6.7 | 7.0 | 7.5 |
| `server_infer_ms` | 191.0 | 6.3 | 190.0 | 191.7 | 214.4 |
| `client_infer_ms` | 193.9 | 6.4 | 192.9 | 194.6 | 217.6 |
| `server_prev_total_ms` | 195.2 | 9.0 | 193.2 | 195.7 | 239.1 |

吞吐约 **5.15 it/s**（端到端）。

#### 怎么读这些数字（重要更正）

> **不要把 `policy_infer_ms ≈ 6.7 ms` 当成真实模型推理延迟。**  
> 它与「TensorRT ~90 ms」一类数字**不可比**；更接近真实墙钟的是 **`server_infer_ms ≈ 191 ms`**。

1. **`policy_infer_ms ≈ 6.7 ms` 是 JAX 异步计时假象**  
   代码在 `Policy.infer`（`src/openpi/policies/policy.py`）里大致是：

   ```text
   t0 = now()
   actions = sample_actions(...)   # 只把计算 enqueue 到 GPU，通常不阻塞
   policy_infer_ms = now() - t0    # ← 停表（此时 GPU 往往还没算完）
   np.asarray(actions)             # ← 这里才 device→host 同步，真正等 GPU
   ```

   所以 6.7 ms 主要是 **dispatch / 启动开销**，方差极小也符合「几乎没在等 GPU」的特征。

2. **`server_infer_ms ≈ 191 ms` 才是当前更可信的单次请求墙钟**  
   它包住整次 `policy.infer()`（含上面的 `np.asarray` 同步），因此把真实 GPU 时间算进去了。此外还有 input/output transform、（若走 WM）编排等。相对「TRT 加速后 ~90 ms」，本机 **~190 ms 量级**更合理；JAX eager serving 通常也慢于专门 TRT 引擎。

3. **`client_infer_ms ≈ server_infer_ms`**  
   本机 `127.0.0.1`，网络可忽略（约 194 vs 191 ms）。

4. **和论文 / FAAC / TRT 怎么比**  
   - 本节只证明：**链路已通**（权重加载 + websocket + 能出 action）。  
   - **比延迟**：看 `server_infer_ms` / `client_infer_ms`，或给 `sample_actions` 后加 `jax.block_until_ready(...)` 再停表，得到诚实的 `policy_infer_ms`。  
   - **比 FAAC**：仍应用 **SR–\(\Delta\)**，不要拿未同步的 6.7 ms 去论证「异步几乎不用掩盖」。  
   - 与 TRT ~90 ms 对比时，先对齐：同一模型/分辨率/步数、是否含预处理、是否已 warmup、是否 `block_until_ready`。

5. **后续建议**  
   - 若要修正计时：在停表前对 actions 做 `jax.block_until_ready`（或把 `np.asarray` 移入计时段）。  
   - 正式对比前 `nvidia-smi` 清无关 GPU 进程，保持 `XLA_*` 设置。  
   - 继续 §8 冒烟与扫 \(\Delta\)。

---



## 8. LIBERO 异步评估（验证 FAAC 加速可用性）

异步参数约定（默认 H=10）：


| 符号               | 含义                 | 脚本变量        |
| ---------------- | ------------------ | ----------- |
| H                | action horizon     | `AH`（默认 10） |
| K                | 异步触发步              | `K`（默认 9）   |
| \Delta / overlap | 执行与预测错位，\Delta=H-K | `OVERLAP`   |


\Delta 越大，模拟的推理延迟越大；论文关心的是大 \Delta 下成功率是否仍高。

### 8.1 冒烟（先确认能跑通）

> 脚本会**自己**起 `serve_policy`；请先停掉占用 `PORT` 的旧 server（例如 §7 延迟压测留下的进程）。  
> 必须先 export §6 的路径与 `XLA_*`（脚本强制要求 `PI0_CHECKPOINT` / `PY_SERVER` / `WM`）。

```bash
cd /home/nvidia/stephen/01-code/Jetson-PI
source .venv/bin/activate

# —— §6 必填（冒烟/正式评估共用）——
export PYTHONNOUSERSITE=1
export REPO="$(pwd)"
export PY_SERVER="${REPO}/.venv/bin/python"
export PI0_CHECKPOINT="${REPO}/checkpoints/jetson-pi-pi05/pi05_libero"
export WM="${REPO}/checkpoints/jetson-pi-pi05/future_correction_module"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

# —— 冒烟参数 ——
export LIBERO_WM_EVAL_NUM_TRIALS=2          # 冒烟用；正式对比改回 50
export LIBERO_WM_EVAL_TASK_SUITE=libero_spatial
export AH=10
export K=9                                   # Δ=1
export OVERLAP=$((AH - K))
export PORT=8000

# 若 8000 仍被占用：
#   fuser -k ${PORT}/tcp   # 或 Ctrl-C 掉旧 serve_policy

bash scripts/eval_wm_libero_spatial.sh
```

输出目录：`logs/<suite>_*_h${AH}_k${K}_*/`

- `client.log`：成功率、任务进度
- `serve.log`：server 日志
- `run_meta.txt`：本次参数
- `videos/`：回放



### 8.2 正式单点（FAAC，对应论文 Ours）

先 export §8.1 中的 §6 必填变量，再：

```bash
export LIBERO_WM_EVAL_NUM_TRIALS=50
export LIBERO_WM_EVAL_TASK_SUITE=libero_spatial
export AH=10
export K=9          # 改 K 即可扫不同 Δ；例如 K=5 → Δ=5
export OVERLAP=$((AH - K))
export PORT=8000

bash scripts/eval_wm_libero_spatial.sh
```

扫多个 \Delta 示例：

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
2. **算法侧加速收益**：在第 8 节增大 \Delta 时看成功率
  - 基线异步方法（VLASH/RTC）在大 \Delta 时 SR 掉得快  
  - FAAC（Ours / +Sched）应更接近同步水平（见 README 表格）
3. **对照论文**：同一模型扫 \Delta=1..9，对比 `libero_spatial/object/goal/10` 的 SR。
  - 论文表格里的 Orin 延迟比例与 Thor 实测延迟不同；**SR–\Delta 曲线**仍可对照，**绝对 ms** 请单独记录为 Thor 结果。

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


| 侧    | 版本                                                     |
| ---- | ------------------------------------------------------ |
| 本机   | JetPack 7.2 / L4T R39 / CUDA **13.2** / GPU **sm_110** |
| 仓库默认 | `jax[cuda12]==0.5.3`（CUDA 12.x wheel，无 Blackwell 支持）   |


结果通常是：装得上但 kernel 不可用、编译失败、或落到错误 backend。这不是漏步骤，而是 **默认依赖面向 Orin/CUDA12**。

### 12.2 推荐出路


| 方案                                    | 适用                                  |
| ------------------------------------- | ----------------------------------- |
| **A.** `jax[cuda13]` **≥0.10（本文 §3）** | 本机直接跑 `serve_policy` / LIBERO（优先尝试） |
| B. jetson-containers / NGC 预编译 JAX    | PyPI wheel 有 `no kernel image` 等错误时 |
| C. 换 x86 + 大显存 CUDA 机器跑评估             | 只关心算法 SR，不关心 Thor 延迟数字              |
| D. 板端走 Jetson-PI-Edge                 | 测 llama.cpp 引擎，而非 JAX               |




### 12.3 验证仍失败时

- 确认 `uv pip show jax jaxlib` 是 **cuda13** 插件，而不是残留的 `jax-cuda12-`* / `0.5.3`。
- 确认 `nvidia-smi` 能看到 Thor，且无其他进程占满内存。
- 删除重建 `.venv`，避免混用 cuda12 / cuda13 plugin。
- 查看是否误设 `CUDA_VISIBLE_DEVICES=""` 或 `JAX_PLATFORMS=cpu`。

---



## 13. 其他常见问题


| 现象                                                         | 处理                                                                          |
| ---------------------------------------------------------- | --------------------------------------------------------------------------- |
| `no kernel image is available for execution on the device` | JAX 未含 sm_110；换 §3 的 cuda13 / 0.10+ 或 jetson-containers 包                   |
| `cudaErrorInsufficientDriver`                              | 在 Thor JP7.2 上少见；检查是否用了错误容器/旧 `LD_LIBRARY_PATH`                             |
| `No module named 'jax'`                                    | `PY_SERVER` 必须指向 `.venv/bin/python`                                         |
| `No solution ... torch==1.11.0+cu113` | 按 §4 过滤 torch 行，再装 CPU 或 jetson-ai-lab cu130 |
| `llvmlite` / `Cannot install on Python 3.11` | 旧 pin 仅支持 &lt;3.10。按 §4 预装 `numba>=0.59` + `--override` |
| `third_party/libero` 为空 / 装不上 | `git submodule update --init --recursive` |
| `No module named 'bddl'` | `third_party/libero` 用了 `--no-deps`。在 **libero client** venv：`uv pip install --python examples/libero/.venv/bin/python 'bddl==1.0.1'`，并确认已 `--no-editable` 安装 libero |
| `Weights only load failed` / `torch.load` UnpicklingError | PyTorch≥2.6 默认 `weights_only=True`。本仓库已改 `third_party/libero/.../benchmark/__init__.py` 对 init states 用 `weights_only=False`；若 submodule 被重置需重新打补丁 |
| `No module named 'tyro'` | 用的是缺依赖的根 `.venv`；按 §7.0 补齐，且 `PY_SERVER` 必须指向仓库根 `.venv/bin/python` |
| `DeviceLocalLayout` / `jax.experimental.layout` | `orbax-checkpoint==0.11.13` 与 `jax 0.10` 不兼容。`cd /tmp && uv pip install --python .../.venv/bin/python -U 'orbax-checkpoint>=0.11.25'` |
| `StepMetadata object is not subscriptable` | orbax≥0.12 的 metadata API 变化；本仓库已在 `restore_params` 兼容 `item_metadata`。拉取最新代码后重试 |
| `_CHECKPOINT_METADATA` does not exist | 警告可忽略（本机权重仍可读） |
| `concrete` option to `jax.checkpoint` removed | `flax==0.10.2` 不支持 jax 0.10。升级：`uv pip install 'flax>=0.12' --index-url https://pypi.org/simple`（清华源可能无新版本） |
| `rich` has no attribute `table` | rich 新版本不再支持 `rich.table.Table` 写法；本仓库 `examples/simple_client/main.py` 已改为 `from rich.table import Table`。重跑 client 即可 |
| `jnp.ones` / matmul 后整机卡死 | **立刻停试**；确认已设 `XLA_PYTHON_CLIENT_PREALLOCATE=false` + `MEM_FRACTION≤0.3`；另开 SSH `pkill`；仍挂则换 §3.2 jetson-containers/NGC JAX，勿用交互式裸跑 GPU |
| OOM / 与其他进程抢 GPU                                           | `nvidia-smi` 停掉无关推理；`XLA_PYTHON_CLIENT_PREALLOCATE=false`；降低 `MEM_FRACTION` |
| 缺 `norm_stats`                                             | `PI0_CHECKPOINT` 指到含 `assets/.../norm_stats.json` 的 `pi05_libero`           |
| LIBERO / EGL 报错                                            | 装 `xvfb`，或 `MUJOCO_GL=egl` / `glx`                                          |
| Client 装不上旧 torch pin                                      | 见 §4，改用 Jetson/Thor 可用的 PyTorch                                             |
| GitHub TLS / GnuTLS -110                                   | `git config --global http.version HTTP/1.1` 后重试                             |
| port 占用                                                    | 换 `PORT`，或 `fuser -k ${PORT}/tcp`                                           |
| `third_party/libero` 不完整                                   | `git submodule update --init --recursive`                                   |


---



## 14. 推荐执行顺序（Thor checklist）

- [x] `git submodule update --init --recursive`
- [x] 确认 `R39` + CUDA 13 + `nvidia-smi` 可见 Thor；`nvpmodel` 为 MAXN（或目标功耗档）
- [x] **配置镜像（§1.1）**：PyPI→清华；`uv` 装 Python→南大
- [x] 安装 `uv`；用 Python 3.11 建 `.venv`
- [x] **按 §3 / §7.0 安装** `jax[cuda13]==0.10.2` + 兼容 `flax`/`orbax`（非仓库默认 0.5.3）
- [x] ModelScope 下载 `Jetson-PI-pi05`（结构已核对）
- [x] **`simple_client` 测延迟已跑通**（§7.3：墙钟约 **`server_infer_ms` ~191 ms**；`policy_infer_ms` 6.7 ms 为 JAX 未同步假象，见解读）
- [ ] `LIBERO_WM_EVAL_NUM_TRIALS=2` 冒烟
- [ ] `NUM_TRIALS=50`，扫多个 \(\Delta\)（改 `K`）得到 SR 曲线
- [ ] （可选）开 `LIBERO_WM_EVAL_ADAPTIVE_KAPPA=1` 对比 +Sched
- [ ] （后续）评估 Jetson-PI-Edge 是否支持 Thor，再测板端引擎加速
