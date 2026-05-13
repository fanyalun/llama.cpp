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
  --attention-backend gpu \
  -ngl auto \
  --flash-attn on \
  --lengths 8192,16384,32768,65536,131072 \
  --decode-tokens 16 \
  --repeats 5 \
  --out-dir results/qwen3vl_moe_bench
```

This keeps routed expert weights on CPU through MoE offload while placing KV
cache and Attention on GPU. FlashAttention uses the llama.cpp/ggml backend
implementation; no external Python `flash-attn` package is required.

Smoke test:

```sh
./build/bin/llama-qwen3vl-moe-bench \
  -m /path/to/Qwen3-VL-30B-A3B-Instruct.gguf \
  --mode moe_cpu_offload \
  --attention-backend gpu \
  -ngl auto \
  --flash-attn on \
  --lengths 8192 \
  --decode-tokens 1 \
  --repeats 1 \
  --out-dir results/qwen3vl_moe_bench_smoke
```

CPU-only fallback:

```sh
./build/bin/llama-qwen3vl-moe-bench \
  -m /path/to/model.gguf \
  --mode cpu_full \
  --lengths 8192,16384,32768,65536,131072 \
  --decode-tokens 16 \
  --out-dir results/qwen3vl_moe_bench_cpu
```

Generate plots:

```sh
python3 tools/qwen3vl-moe-bench/plot_qwen3vl_moe_bench.py \
  --summary results/qwen3vl_moe_bench/summary.csv \
  --out-dir results/qwen3vl_moe_bench/plots
```

Outputs:

- `results.jsonl`: raw node timings, per-layer attention summaries, runtime MoE copy records, pinned H2D copy records, and microbench records.
- `summary.csv`: aggregated averages and standard deviations. Microbench rows use `mode=micro_hN` for the active expert-count sweep.
- `plots/figure1_attention_time.svg`: prefill/decode average per-layer Attention time.
- `plots/figure2_attention_copy_ratio.svg`: average per-layer Attention time divided by one routed expert pinned H2D copy time.
- `plots/figure3_moe_gemm.svg`: Serial vs Group GEMM full expert FFN sweep for `h=1..8`.
- `report.md`: command and configuration snapshot.
