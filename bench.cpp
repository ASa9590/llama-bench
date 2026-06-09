#include "llama.h"
#include <cstdio>
#include <string>
#include <vector>
#include <cstring>
struct BenchResult {
    std::string model_name;
    float model_size_mib;
    float load_time_ms;
    float prompt_speed;      // tokens/s, prompt processing
    float generation_speed;  // tokens/s, token generation
};

BenchResult run_benchmark(const std::string & model_path, const std::string & prompt,  int n_gpu_layers) {



    BenchResult result;
    result.model_name = model_path;

    llama_log_set([](enum ggml_log_level, const char *, void *){}, nullptr);


    ggml_backend_load_all();

    // --- 1. 加载模型 ---
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
 

    int64_t t_load_start = ggml_time_ms();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    result.load_time_ms = (float)(ggml_time_ms() - t_load_start);

    if (!model) {
        fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
        return result;
    }

    result.model_size_mib = (float)llama_model_size(model) / 1024.0f / 1024.0f;

    // --- 2. 分词 ---
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 先用 NULL 试一次，返回值的负数就是所需 token 数量
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), n_prompt, true, true);

    // --- 3. 初始化推理上下文 ---
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = n_prompt + 128; // context 大小 = prompt + 生成长度
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return result;
    }

    // --- 4. 初始化采样器（greedy：每次选概率最高的 token）---
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // --- 5. 推理循环 ---
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    int64_t t_gen_start = 0;
    int     n_decode    = 0;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + 128; ) {
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "decode failed\n");
            break;
        }
        n_pos += batch.n_tokens;

        // 第一次 decode 处理的是 prompt，之后才是生成
        if (n_pos >= n_prompt && t_gen_start == 0) {
            t_gen_start = ggml_time_us();
        }

        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) break;

        batch = llama_batch_get_one(&new_token, 1);
        n_decode++;
    }

    int64_t t_gen_end = ggml_time_us();

    // --- 6. 从 llama_perf 取 prompt 速度，自己算生成速度 ---
    llama_perf_context_data perf = llama_perf_context(ctx);
    // t_p_eval_ms: prompt 处理耗时（毫秒），n_p_eval: prompt token 数
    result.prompt_speed     = (float)(perf.n_p_eval) / (perf.t_p_eval_ms / 1000.0f);
    result.generation_speed = (float)n_decode / ((t_gen_end - t_gen_start) / 1e6f);

    // --- 7. 释放资源（RAII 的手动版本）---
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return result;
}
BenchResult run_benchmark(const std::string & model_path, const std::string & prompt, int n_gpu_layers);

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [model2.gguf ...]\n", argv[0]);
        return 1;
    }

    int n_gpu_layers = 99;
    int i = 1;

    // 检测 --cpu 参数
    if (strcmp(argv[i], "--cpu") == 0) {
        n_gpu_layers = 0;
        i++;
    }

    const std::string prompt = "介绍一下深度学习的基本原理";

    printf("%-50s %10s %12s %12s %12s\n",
           "model", "size(MiB)", "load(ms)", "prompt(t/s)", "gen(t/s)");
    printf("%s\n", std::string(100, '-').c_str());

    for (; i < argc; i++) {
        BenchResult r = run_benchmark(argv[i], prompt, n_gpu_layers);
        printf("%-50s %10.1f %12.1f %12.1f %12.1f\n",
               r.model_name.c_str(),
               r.model_size_mib,
               r.load_time_ms,
               r.prompt_speed,
               r.generation_speed);
    }

    return 0;
}