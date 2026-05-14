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

Then combine the existing Attention summary with the expert-copy summary for
the ratio plot:

```sh
python3 tools/qwen3vl-moe-bench/plot_qwen3vl_moe_bench.py \
  --summary results/qwen3vl_attn_placement/summary.csv \
  --expert-summary results/qwen3vl_expert_copy/summary.csv \
  --out-dir results/qwen3vl_attn_placement/plots
```

Outputs:

- `results.jsonl`: raw node timings, `attention_kv_h2d` records, per-layer attention summaries, runtime MoE copy records, `expert_h2d_pinned` records, and microbench records.
- `summary.csv`: aggregated averages and standard deviations. Decode `attention_layer` rows are per-token averages. Microbench rows use `mode=micro_hN` for the active expert-count sweep.
- `plots/figure1_attention_time.svg`: prefill/decode average per-layer Attention time.
- `plots/figure2_attention_expert_copy_ratio.svg`: `KV CPU + Attn CPU` average per-layer Attention time divided by one routed expert pinned H2D copy time.
- `plots/figure3_moe_gemm.svg`: Serial vs Group GEMM full expert FFN sweep for `h=1..8`.
- `report.md`: command and configuration snapshot.
