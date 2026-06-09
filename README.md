# llama-bench
基于 `llama.cpp` 的轻量级大模型基准测试工具

通过本项目，你可以方便地测试不同的 GGUF 格式大模型在当前硬件下的**加载速度**、**Prompt 处理速度（Prefill）**以及**文本生成速度（Generation）**。

## 功能特点
1. 纯C++实现，直接调用`llama.cpp`底层API，无多余依赖
2. 核心评价指标
    size(MiB): 展示模型的实际占用内存/显存大小
    load(ms): 精确记录模型从磁盘加载到内存/显存的耗时
    Prompt(t/s): 评估Prompt的突吞吐量(每秒处理的Token数)
    gen(t/s): 评估文本生成的速率(美秒生成的Token数)
3. 硬件切换灵活，默认开启GPU全层卸载，同时支持通过参数切换为纯CPU推理

## 环境准备

本项目依赖于`llama.cpp`库（支持纯 CPU 或 CUDA 硬件加速），且编译需要 CMake 3.18+。


### 下载模型

如果本地没有 GGUF 模型，可以使用 huggingface_hub 快速下载一个轻量级模型用于测试：

pip install huggingface_hub
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir ~/models/

## 项目编译

在终端中进入项目根目录(包含`CMakeLists.txt`)，依次执行：

1. 创建并进入编译目录
    mkdir build && cd build
2. 配置CMake
    * CPU模式
    cmake ..

    * GPU/cuda模式，如果有 NVIDIA 显卡，请使用此命令开启硬件加速
    cmake -DGGML_CUDA=ON ..
3. 执行编译
    cmake --build . --config Release

编译完成后，当前 build 目录下会生成可执行文件 llama-bench-simple。

## 运行指南
以下命令均在 build 目录下执行。

### GPU 推理（默认，自动卸载 99 层到显存）

./llama-bench-simple ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf

### 纯CPU推理

./llama-bench-simple --cpu ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf

### 批量对比多个模型
./llama-bench-simple model_q4.gguf model_q8.gguf model_f16.gguf

## 实验结论

测试环境：WSL2 Ubuntu 20.04，NVIDIA GPU（CUDA 12.8），提示词："介绍一下深度学习的基本原理"

| 运行模式 | 模型大小 (MiB) | 加载耗时 (ms) | Prompt 速度 (t/s) | 生成速度 (t/s) |
|----------|--------------|-------------|-----------------|--------------|
| GPU Q4_K_M | 1059.9 | 2779 | 106 | 217 |
| GPU Q8_0   | 1801.1 | 4783 | 234 | 164 |
| CPU Q4_K_M | 1059.9 | 553  | 60  | 35  |
| CPU Q8_0   | 1801.1 | 4094 | 31  | 22  |

## 结论

**1. GPU 对生成速度有决定性影响**
GPU Q4_K_M 生成速度 217 t/s，CPU Q4_K_M 仅 35 t/s，GPU **快 6.2 倍**。
Token 生成是内存带宽瓶颈（每步只读一次权重），GPU 显存带宽远高于内存带宽，优势明显。

**2. Prompt 处理阶段，Q8_0 反而比 Q4_K_M 快**
GPU 上 Q8_0 prompt 速度 234 t/s，Q4_K_M 只有 106 t/s。
原因：Prompt 阶段是计算密集型，Q4 需要额外反量化开销；Q8_0 格式更简单，GPU 并行计算效率更高。

**3. 量化对生成速度影响有限，但显著节省显存**
GPU 上 Q4 生成速度（217）比 Q8 （164）快约 1.3 倍，但模型体积缩小 41%（1060 vs 1801 MiB）。
在显存有限的边缘设备上，Q4 是性价比最优选择。