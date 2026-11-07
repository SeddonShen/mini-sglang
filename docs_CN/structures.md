# Mini-SGLang 结构说明

## 系统架构

Mini-SGLang 被设计为一个分布式系统，以高效处理大语言模型（LLM）推理。它由多个协同工作的独立进程组成。

### 核心组件

- **API 服务器（API Server）**：用户入口。提供 OpenAI 兼容的 API（如 `/v1/chat/completions`），接收提示词并返回生成的文本。
- **分词器进程（Tokenizer Worker）**：将输入文本转换为模型能理解的数字（token）。
- **反分词器进程（Detokenizer Worker）**：将模型生成的数字（token）转换回人类可读的文本。
- **调度器进程（Scheduler Worker）**：核心工作进程。在多 GPU 部署中，每块 GPU 对应一个调度器进程（称为 **TP Rank**）。它负责管理该 GPU 上的计算与资源分配。

### 数据流

各组件之间使用 **ZeroMQ（ZMQ）** 传递控制消息，使用 **NCCL**（通过 `torch.distributed`）在 GPU 之间交换大型张量数据。

![进程总览图](https://lmsys.org/images/blog/minisgl/design.drawio.png)

**请求的生命周期：**

1. **用户**向 **API 服务器** 发送请求。
2. **API 服务器**将其转发给 **分词器（Tokenizer）**。
3. **分词器**将文本转为 token，发送给 **调度器（Rank 0）**。
4. **调度器（Rank 0）**将请求广播给其他所有调度器（多 GPU 场景下）。
5. **所有调度器**对请求进行调度，并触发各自的本地 **引擎（Engine）** 计算下一个 token。
6. **调度器（Rank 0）**收集输出的 token，发送给 **反分词器（Detokenizer）**。
7. **反分词器**将 token 转为文本，回传给 **API 服务器**。
8. **API 服务器**将结果流式返回给 **用户**。

## 代码组织（`minisgl` 包）

源码位于 `python/minisgl`。以下是为开发者整理的模块说明：

- `minisgl.core`：提供核心数据类 `Req` 和 `Batch`（表示请求状态）、`Context` 类（持有推理上下文的全局状态），以及 `SamplingParams` 类（持有用户提供的采样参数）。
- `minisgl.distributed`：提供张量并行中的 all-reduce 与 all-gather 接口，以及持有 TP worker 张量并行信息的 `DistributedInfo` 数据类。
- `minisgl.layers`：实现构建支持 TP 的 LLM 所需的基础构件，包括线性层、层归一化（layernorm）、嵌入（embedding）、RoPE 等。它们共享定义在 `minisgl.layers.base` 中的公共基类。
- `minisgl.models`：实现 LLM 模型，包括 Llama 和 Qwen3。同时定义了从 HuggingFace 加载权重以及权重切分的工具。
- `minisgl.attention`：提供注意力后端的接口，并实现了 `flashattention` 和 `flashinfer` 后端。它们由 `AttentionLayer` 调用，并使用 `Context` 中存储的元数据。
- `minisgl.kvcache`：提供 KVCache 池和 KVCache 管理器的接口，并实现了 `MHAKVCache`、`NaiveCacheManager` 和 `RadixCacheManager`。
- `minisgl.utils`：提供一组工具函数，包括日志配置和 zmq 的封装。
- `minisgl.engine`：实现 `Engine` 类——单进程上的一个 TP worker。它管理模型、上下文、KVCache、注意力后端以及 CUDA Graph 的重放。
- `minisgl.message`：定义 api_server、tokenizer、detokenizer 与 scheduler 之间（经 zmq）交换的消息。所有消息类型都支持自动序列化与反序列化。
- `minisgl.scheduler`：实现 `Scheduler` 类，运行在每个 TP worker 进程上，管理对应的 `Engine`。Rank 0 的调度器从 tokenizer 接收消息、与其他 TP worker 上的调度器通信，并向 detokenizer 发送消息。
- `minisgl.server`：定义命令行参数和 `launch_server`（用于启动 Mini-SGLang 的所有子进程）。同时在 `minisgl.server.api_server` 中实现了一个作为前端的 FastAPI 服务器，提供 `/v1/chat/completions` 等端点。
- `minisgl.tokenizer`：实现 `tokenize_worker` 函数，处理分词与反分词请求。
- `minisgl.llm`：提供 `LLM` 类，作为与 Mini-SGLang 系统交互的 Python 接口。
- `minisgl.kernel`：实现自定义 CUDA 算子，由 `tvm-ffi` 提供 Python 绑定和 JIT 接口。
- `minisgl.benchmark`：性能测试工具。
