<p align="center">
<img width="400" src="/assets/logo.png">
</p>

# Mini-SGLang（中文版）

一个**轻量且高性能**的大语言模型（LLM）推理框架。

---

Mini-SGLang 是 [SGLang](https://github.com/sgl-project/sglang) 的精简实现，旨在揭开现代 LLM 服务系统的复杂性。它仅用 **约 5,000 行 Python 代码**，既是一个可用的推理引擎，也是面向研究者和开发者的透明参考实现。

## ✨ 核心特性

- **高性能**：借助先进的优化技术，达到业界领先的吞吐量与延迟。
- **轻量且易读**：代码库干净、模块化、带完整类型标注，易于理解与修改。
- **先进优化**：
  - **Radix Cache（基数树缓存）**：跨请求复用共享前缀的 KV 缓存。
  - **Chunked Prefill（分块预填充）**：降低长上下文服务的峰值显存占用。
  - **Overlap Scheduling（重叠调度）**：用 GPU 计算掩盖 CPU 调度开销。
  - **Tensor Parallelism（张量并行）**：跨多 GPU 扩展推理。
  - **优化算子**：集成 **FlashAttention** 与 **FlashInfer** 以获得最高效率。
  - ...

## 🚀 快速开始

> **⚠️ 平台支持**：Mini-SGLang 目前**仅支持 Linux**（x86_64 与 aarch64）。由于依赖 Linux 特有的 CUDA 算子（`sgl-kernel`、`flashinfer`），不支持 Windows 和 macOS。Windows 用户建议使用 [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)，或使用 Docker 实现跨平台兼容。

### 1. 环境准备

推荐使用 `uv` 进行快速可靠的安装（注意 `uv` 与 `conda` 不冲突）。

```bash
# 创建虚拟环境（推荐 Python 3.10+）
uv venv --python=3.12
source .venv/bin/activate
```

**前置要求**：Mini-SGLang 依赖 JIT 编译的 CUDA 算子。请确保已安装 **NVIDIA CUDA Toolkit**，且其版本与驱动匹配。可用 `nvidia-smi` 查看驱动支持的 CUDA 版本。

### 2. 安装

直接从源码安装 Mini-SGLang：

```bash
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
```

<details>
<summary><b>💡 在 Windows (WSL2) 上安装</b></summary>

由于 Mini-SGLang 依赖 Linux 特有的组件，Windows 用户应使用 WSL2：

1. **安装 WSL2**（如果尚未安装）：
   ```powershell
   # 在 PowerShell（管理员）中执行
   wsl --install
   ```

2. **在 WSL2 中安装 CUDA**：
   - 参照 [NVIDIA 的 WSL2 CUDA 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   - 确保 Windows 的 GPU 驱动支持 WSL2

3. **在 WSL2 中安装 Mini-SGLang**：
   ```bash
   # 在 WSL2 终端内执行
   git clone https://github.com/sgl-project/mini-sglang.git
   cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
   uv pip install -e .
   ```

4. **从 Windows 访问**：在 Windows 的浏览器和应用程序中可通过 `http://localhost:8000` 访问服务。

</details>

<details>
<summary><b>🐳 使用 Docker 运行</b></summary>

**前置要求**：
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

1. **构建 Docker 镜像**：
   ```bash
   docker build -t minisgl .
   ```

2. **启动服务**：
   ```bash
   docker run --gpus all -p 1919:1919 \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

3. **以交互式终端模式运行**：
   ```bash
   docker run -it --gpus all \
       minisgl --model Qwen/Qwen3-0.6B --shell
   ```

4. **使用 Docker 卷持久化缓存**（推荐，可加快后续启动速度）：
   ```bash
   docker run --gpus all -p 1919:1919 \
       -v huggingface_cache:/app/.cache/huggingface \
       -v tvm_cache:/app/.cache/tvm-ffi \
       -v flashinfer_cache:/app/.cache/flashinfer \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

</details>

### 3. 在线服务

一条命令即可启动 OpenAI 兼容的 API 服务。

```bash
# 在单 GPU 上部署 Qwen/Qwen3-0.6B
python -m minisgl --model "Qwen/Qwen3-0.6B"

# 使用张量并行在 4 GPU 上部署 meta-llama/Llama-3.1-70B-Instruct，端口 30000
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000
```

服务启动后，即可用 `curl` 等标准工具或任何 OpenAI 兼容客户端发送请求。

### 4. 交互式终端

加上 `--shell` 参数，即可直接在终端里与模型对话。

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

![shell-example](https://lmsys.org/images/blog/minisgl/shell.png)

也可以使用 `/reset` 清空对话历史。

## 性能测试

### 离线推理

详见 [bench.py](./benchmark/offline/bench.py)。如需对重叠调度做消融实验，可设置 `MINISGL_DISABLE_OVERLAP_SCHEDULING=1`。

测试配置：

- 硬件：1xH200 GPU。
- 模型：Qwen3-0.6B、Qwen3-14B
- 总请求数：256 条序列
- 输入长度：在 100-1024 个 token 之间随机采样
- 输出长度：在 100-1024 个 token 之间随机采样

![offline](https://lmsys.org/images/blog/minisgl/offline.png)

### 在线推理

详见 [benchmark_qwen.py](./benchmark/online/bench_qwen.py)。

测试配置：

- 硬件：4xH200 GPU，NVLink 互联。
- 模型：Qwen3-32B
- 数据集：[Qwen trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/blob/main/qwen_traceA_blksz_16.jsonl)，回放前 1000 条请求。

启动命令：

```bash
# Mini-SGLang
python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --cache naive

# SGLang
python3 -m sglang.launch_server --model "Qwen/Qwen3-32B" --tp 4 \
    --disable-radix --port 1919 --decode-attention flashinfer
```

> **注**：如果从 HuggingFace 下载模型遇到网络问题，可尝试 `--model-source modelscope` 改从魔搭（ModelScope）下载：
> ```bash
> python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --model-source modelscope
> ```

![online](https://lmsys.org/images/blog/minisgl/online.png)

## 📚 了解更多

- **[特性详解](./docs_CN/features.md)**：浏览所有可用特性与命令行参数。
- **[系统架构](./docs_CN/structures.md)**：深入了解 Mini-SGLang 的设计与数据流。
