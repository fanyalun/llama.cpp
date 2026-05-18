# Qwen3-VL MoE Benchmark

This tool targets Qwen3-VL MoE GGUF experiments where the GPU cannot hold all
experts.

Build:

```sh
cmake -B build -DGGML_CUDA=ON
cmake --build build --target llama-qwen3vl-moe-bench -j
```

Recommended run:

```sh
./build/bin/llama-qwen3vl-moe-bench \
  -m /path/to/Qwen3-VL-30B-A3B-Instruct.gguf \
  --mode moe_cpu_offload \
  --attention-placement all \
  -ngl auto \
  --flash-attn on \
  --lengths 1024,4096,8192,16384 \
  --decode-tokens 16 \
  --repeats 3 \
  --skip-microbench \
  --out-dir results/qwen3vl_attn_placement
```

`--attention-placement all` runs:

- `kv_gpu_attn_gpu`: KV cache and Attention on GPU.
- `kv_cpu_attn_cpu`: KV cache and Attention on CPU.
- `kv_cpu_attn_gpu`: KV cache stored on CPU, copied temporarily to GPU for Attention.

For `kv_cpu_attn_gpu`, per-layer Attention time includes CPU KV cache to GPU
copy time plus GPU Attention compute time. FlashAttention uses the
llama.cpp/ggml backend implementation; no external Python `flash-attn` package
is required.

Smoke test:

```sh
./build/bin/llama-qwen3vl-moe-bench \
  -m /path/to/Qwen3-VL-30B-A3B-Instruct.gguf \
  --mode moe_cpu_offload \
  --attention-placement all \
  -ngl auto \
  --flash-attn on \
  --lengths 1024 \
  --decode-tokens 1 \
  --repeats 1 \
  --skip-microbench \
  --out-dir results/qwen3vl_moe_bench_smoke
```

CPU-only fallback:

```sh
./build/bin/llama-qwen3vl-moe-bench \
  -m /path/to/model.gguf \
  --mode cpu_full \
  --lengths 1024,4096,8192,16384 \
  --decode-tokens 16 \
  --out-dir results/qwen3vl_moe_bench_cpu
```

Generate plots:

```sh
python3 tools/qwen3vl-moe-bench/plot_qwen3vl_moe_bench.py \
  --summary results/qwen3vl_moe_bench/summary.csv \
  --out-dir results/qwen3vl_moe_bench/plots
```

Measure one routed expert copy from CPU pinned memory to GPU without rerunning
the model benchmark:

```sh
./build/bin/llama-qwen3vl-moe-bench \
  -m /path/to/Qwen3-VL-30B-A3B-Instruct.gguf \
  --skip-model-bench \
  --skip-gemm-microbench \
  --repeats 100 \
  -ngl auto \
  --out-dir results/qwen3vl_expert_copy
```

Measure the Alpha MoE GEMM sweep without rerunning the model benchmark:

```sh
./build/bin/llama-qwen3vl-moe-bench \
  -m /path/to/Qwen3-VL-30B-A3B-Instruct.gguf \
  --skip-model-bench \
  --skip-gemm-microbench 0 \
  --micro-tokens 1024,2048,4096,8192,16384,32768 \
  --micro-alpha-pcts 0,25,50,75,100 \
  --repeats 3 \
  -ngl auto \
  --out-dir results/qwen3vl_moe_gemm
```

The Alpha sweep treats `alpha` as the expert-cache hit rate. Prefix experts by
expert id are assumed to already be resident on the GPU and are computed as one
routed `ggml_mul_mat_id` group. The remaining suffix experts are cache misses:
the benchmark measures one pinned-host-to-GPU full-expert weight load per miss
and one serial `ggml_mul_mat_id` compute per miss, then models a two-stream
pipeline where expert loads overlap with compute. The reported `moe_gemm`
`time_us` is the simulated pipeline makespan, including uncovered load bubbles.
Additional summary rows expose `moe_gemm_group_compute`,
`moe_gemm_serial_compute`, `moe_gemm_weight_load`, and
`moe_gemm_pipeline_bubble`.

The same GEMM microbench also records a fixed-cache-hit-rate strategy
comparison. Set `--micro-cache-hit-rate` (default `0.5`) to compare:

- `serial_pipeline`: all active experts are computed serially while miss expert
  weights load in a pipeline.
- `group_wait`: miss expert weights are fully loaded first, then all active
  experts are computed as one routed group.
- `hybrid_pipeline`: cache-hit experts are computed as one routed group while
  miss expert weights load, then miss experts are computed serially in a
  pipeline.

Measure the Alpha MoE GEMM sweep with real layer-10 token-to-expert routing
from the AIME allresults JSON:

```sh
./build/bin/llama-qwen3vl-moe-bench \
  -m /path/to/Qwen3-VL-30B-A3B-Instruct.gguf \
  --skip-model-bench \
  --skip-gemm-microbench 0 \
  --prompt-json benches/dgx-spark/aime25_openai__gpt-oss-120b-high_temp1.0_20251109_094547_allresults.json \
  --route-layer 10 \
  --route-max-prompts 16 \
  --micro-tokens 1024,2048,4096,8192,16384,32768 \
  --micro-alpha-pcts 0,25,50,75,100 \
  --micro-cache-hit-rate 0.5 \
  --repeats 3 \
  -ngl auto \
  --out-dir results/qwen3vl_moe_gemm_real
```

`--prompt-json` changes the prefill GEMM sweep from synthetic long-tail routing
to real `ffn_moe_topk` routing collected at `--route-layer`. Prompt text is read
from each JSON HTML entry's `Prompt conversation` block, tokenized by the loaded
model, packed by concatenating multiple prompts with blank-line separators, and
clipped to each `--micro-tokens` length. Each packed sample starts from a
different prompt offset. Use `--route-max-prompts 0` to collect one packed sample
starting at every prompt for each length.

The Alpha sweep splits active experts by expert id order. The first
`floor(active_experts * alpha / 100)` experts are treated as cache hits, and the
remaining experts are treated as cache misses whose weight loads are pipelined
with compute. Prefill uses the model expert count as active experts and
distributes input tokens with a fixed `N,N-1,...,1` long-tail weight after
assigning one token per expert. Decode uses one token with Top-K `8` active
experts and records one row per alpha.

The fixed-cache-hit-rate strategy comparison also uses expert id order:
`floor(active_experts * cache_hit_rate)` prefix experts are cache hits, and the
remaining suffix experts are cache misses.

Then combine the existing Attention summary with the expert-copy summary for
the ratio plot:

```sh
python3 tools/qwen3vl-moe-bench/plot_qwen3vl_moe_bench.py \
  --summary results/qwen3vl_attn_placement/summary.csv \
  --expert-summary results/qwen3vl_expert_copy/summary.csv \
  --out-dir results/qwen3vl_attn_placement/plots
```

Outputs:

- `results.jsonl`: raw node timings, `attention_kv_h2d` records, per-layer attention summaries, runtime MoE copy records, `expert_h2d_pinned` records, route distributions, and microbench records. GEMM records include `group_compute_us`, `serial_compute_us`, `weight_load_us`, and `pipeline_bubble_us`; fixed-cache strategy records additionally include `cache_hit_rate`, `hit_experts`, `miss_experts`, and `strategy`.
- `summary.csv`: aggregated averages and standard deviations. Decode `attention_layer` rows are per-token averages. Alpha GEMM rows use `mode=micro_alphaN`; fixed-cache strategy rows use `mode=micro_cacheN_<strategy>`.
- `plots/figure1_attention_time.svg`: prefill/decode average per-layer Attention time.
- `plots/figure2_attention_expert_copy_ratio.svg`: `KV CPU + Attn CPU` average per-layer Attention time divided by one routed expert pinned H2D copy time.
- `plots/figure3_moe_gemm.svg`: Alpha sweep with Prefill and Decode panels on independent y axes.
- `plots/figure4_moe_cache_strategy.svg`: fixed-cache-hit-rate strategy comparison.
- `report.md`: command and configuration snapshot.
