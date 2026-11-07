# Mini-SGLang 特性说明

## 在线服务

Mini-SGLang 支持通过 OpenAI 兼容的 API 服务器提供在线服务。它提供标准的 `/v1/chat/completions` 端点，可与现有工具和客户端无缝集成。详细的命令行参数与配置选项，请运行 `python -m minisgl --help` 查看。

## 交互式终端模式

为方便演示和测试，框架提供了交互式终端模式。在该模式下，用户可直接输入提示词（prompt），LLM 会实时生成回复。终端会自动缓存对话历史以保持上下文。使用 `/reset` 命令可清空对话历史、开启新会话。

示例：

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

## 分布式服务

为了在多 GPU 上扩展性能，Mini-SGLang 支持张量并行（Tensor Parallelism，TP）。通过 `--tp n` 参数指定 GPU 数量即可启用分布式服务，其中 `n` 为并行度。

## 支持的模型

本框架目前支持以下稠密（dense）模型架构：

- [`Llama-3`](https://huggingface.co/collections/meta-llama/llama-31) 系列
- [`Qwen-3`](https://huggingface.co/collections/Qwen/qwen3) 系列（含 MoE）
- [`Qwen-2.5`](https://huggingface.co/collections/Qwen/qwen25) 系列

## 分块预填充（Chunked Prefill）

分块预填充是 [Sarathi-Serve](https://arxiv.org/abs/2403.02310) 提出的技术，默认开启。该特性在预填充（prefill）阶段将长提示词切分为较小的块，显著降低峰值显存占用，避免长上下文服务中的 OOM（显存不足）错误。分块大小可通过 `--max-prefill-length n` 配置。注意：不建议将 `n` 设得过小（如 128），否则可能明显降低性能。

## 页大小（Page Size）

可通过 `--page-size` 参数指定系统的页大小。

## 注意力后端（Attention Backends）

Mini-SGLang 集成了高性能注意力算子，包括 [`FlashAttention`](https://github.com/Dao-AILab/flash-attention)（`fa`）、[`FlashInfer`](https://github.com/flashinfer-ai/flashinfer)（`fi`）和 [`TensorRT-LLM fmha`](https://github.com/NVIDIA/TensorRT-LLM)（`trtllm`）。它支持为预填充和解码阶段使用不同的后端以最大化效率。例如，在 NVIDIA Hopper GPU 上，默认预填充使用 `FlashAttention 3`、解码使用 `FlashInfer`。

通过 `--attn` 参数指定后端。若提供两个值（如 `--attn fa,fi`），第一个为预填充后端，第二个为解码后端。注意：某些注意力后端可能会覆盖用户指定的页大小（例如 `trtllm` 只支持页大小 16、32、64）。

## CUDA Graph

为减少解码阶段的 CPU 启动开销，Mini-SGLang 支持捕获与重放 CUDA Graph。该特性默认开启。CUDA Graph 捕获的最大批大小可通过 `--cuda-graph-max-bs n` 设置；将 `n` 设为 `0` 可禁用该特性。

## Radix Cache（基数树缓存）

沿用 [SGLang](https://github.com/sgl-project/sglang.git) 的原始设计，Mini-SGLang 实现了 Radix Cache 来管理 KV 缓存。它可以跨请求复用共享前缀的 KV 缓存，减少重复计算。该特性默认开启，也可通过 `--cache naive` 切换为朴素的缓存管理策略。

![radix](https://lmsys.org/images/blog/sglang/radix_attn.jpg)
*Radix Attention 示意图，来自 [LMSYS 博客](https://lmsys.org/blog/2024-01-17-sglang/)。*

## 重叠调度（Overlap Scheduling）

为进一步降低 CPU 开销，Mini-SGLang 采用了 [NanoFlow](https://arxiv.org/abs/2408.12757) 提出的重叠调度技术。该方法将 CPU 调度开销与 GPU 计算重叠执行，提升系统整体吞吐量。

![overlap](https://lmsys.org/images/blog/sglang_v0_4/scheduler.jpg)
*重叠调度示意图，来自 [LMSYS 博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)。*
