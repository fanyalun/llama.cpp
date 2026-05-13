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
  --attention-backend cpu \
  -ngl auto \
  --no-kv-offload \
  --flash-attn on \
  --lengths 8192,16384,32768,65536,131072 \
  --decode-tokens 16 \
  --repeats 5 \
  --out-dir results/qwen3vl_moe_bench
```

GPU-attention run with experts still offloaded to CPU:

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
  --out-dir results/qwen3vl_moe_bench_gpu_attn
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

- `results.jsonl`: raw node timings, per-layer attention summaries, MoE copy records, and microbench records.
- `summary.csv`: aggregated averages and standard deviations.
- `report.md`: command and configuration snapshot.
