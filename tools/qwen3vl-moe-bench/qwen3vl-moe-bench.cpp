#include "arg.h"
#include "common.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <clocale>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

struct bench_params {
    std::string mode = "moe_cpu_offload";
    std::string out_dir = "qwen3vl_moe_bench_results";
    std::vector<int32_t> lengths = { 8192, 16384, 32768, 65536, 131072 };
    std::vector<int32_t> n_cpu_moe_sweep = { 48, 36, 24, 12, 0 };
    int32_t decode_tokens = 16;
    int32_t repeats = 3;
    int32_t n_cpu_moe_layers = -1;
    bool skip_model_bench = false;
    bool skip_microbench = false;
    bool profile_all_nodes = true;
    int32_t micro_experts = 8;
    int32_t micro_d_model = 2048;
    int32_t micro_d_ff = 4096;
    int32_t micro_prefill_tokens = 512;
    int32_t micro_decode_tokens = 1;
    size_t expert_bytes = 0;
};

struct model_info {
    int32_t n_layer = 0;
    int32_t n_embd = 0;
    int32_t n_expert = 0;
    int32_t n_ff_exp = 0;
};

struct node_record {
    std::string mode;
    std::string phase;
    std::string name;
    int32_t seq_len = 0;
    int32_t repeat = 0;
    int32_t layer = -1;
    int64_t time_us = 0;
};

struct copy_record {
    std::string mode;
    std::string phase;
    int32_t seq_len = 0;
    int32_t repeat = 0;
    uint64_t calls = 0;
    uint64_t bytes = 0;
    uint64_t time_us = 0;
};

struct micro_record {
    std::string kind;
    std::string phase;
    int32_t tokens = 0;
    int32_t repeat = 0;
    int64_t time_us = 0;
    uint64_t bytes = 0;
};

struct callback_data {
    bool profile_all_nodes = true;
    std::string mode;
    std::string phase;
    int32_t seq_len = 0;
    int32_t repeat = 0;
    int64_t t_start_us = 0;
    std::vector<node_record> * records = nullptr;
};

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

static std::string csv_escape(const std::string & s) {
    if (s.find_first_of(",\"\n\r") == std::string::npos) {
        return s;
    }
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') {
            out += "\"\"";
        } else {
            out += c;
        }
    }
    out += "\"";
    return out;
}

static std::vector<std::string> split_string(const std::string & s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream iss(s);
    while (std::getline(iss, cur, sep)) {
        if (!cur.empty()) {
            out.push_back(cur);
        }
    }
    return out;
}

static std::vector<int32_t> parse_i32_list(const std::string & s) {
    std::vector<int32_t> out;
    for (const auto & part : split_string(s, ',')) {
        out.push_back(std::stoi(part));
    }
    return out;
}

static bool parse_size_bytes(const std::string & s, size_t & out) {
    char * end = nullptr;
    double value = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || value < 0) {
        return false;
    }

    size_t mul = 1;
    if (*end != '\0') {
        std::string suffix(end);
        std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) { return (char) std::tolower(c); });
        if (suffix == "k" || suffix == "kb") {
            mul = 1024ULL;
        } else if (suffix == "m" || suffix == "mb") {
            mul = 1024ULL*1024ULL;
        } else if (suffix == "g" || suffix == "gb") {
            mul = 1024ULL*1024ULL*1024ULL;
        } else {
            return false;
        }
    }

    out = (size_t) (value * (double) mul);
    return true;
}

static void print_usage(const char * argv0) {
    LOG("\nusage:\n");
    LOG("  %s -m model.gguf [common llama.cpp args] [bench args]\n\n", argv0);
    LOG("bench args:\n");
    LOG("  --mode <cpu_full|moe_cpu_offload|moe_cpu_sweep>  default: moe_cpu_offload\n");
    LOG("  --out-dir <dir>                                  default: qwen3vl_moe_bench_results\n");
    LOG("  --lengths <csv>                                  default: 8192,16384,32768,65536,131072\n");
    LOG("  --decode-tokens <n>                              default: 16\n");
    LOG("  --repeats <n>                                    default: 3\n");
    LOG("  --n-cpu-moe-layers <n>                           for moe_cpu_offload; -1 means all experts\n");
    LOG("  --n-cpu-moe-sweep <csv>                          default: 48,36,24,12,0\n");
    LOG("  --skip-model-bench <0|1>                         default: 0\n");
    LOG("  --skip-microbench <0|1>                          default: 0\n");
    LOG("  --profile-all-nodes <0|1>                        default: 1, exact but slower node timing\n");
    LOG("  --expert-bytes <bytes|MiB|GiB>                   override H2D copy size\n");
    LOG("  --micro-experts <n>                              default: 8\n");
    LOG("  --micro-d-model <n>                              default: model n_embd or 2048\n");
    LOG("  --micro-d-ff <n>                                 default: model expert FFN or 4096\n");
    LOG("  --micro-prefill-tokens <n>                       default: 512\n");
    LOG("  --micro-decode-tokens <n>                        default: 1\n\n");
    LOG("recommended CPU-attention MoE offload run:\n");
    LOG("  %s -m model.gguf --mode moe_cpu_offload --no-kv-offload --flash-attn on -ngl auto --lengths 8192,16384 --decode-tokens 16\n\n", argv0);
}

static bool parse_custom_args(int argc, char ** argv, bench_params & bench, std::vector<std::string> & common_args) {
    common_args.clear();
    common_args.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        try {
            if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                common_args.push_back(arg);
            } else if (arg == "--mode") {
                bench.mode = need_value("--mode");
            } else if (arg == "--out-dir") {
                bench.out_dir = need_value("--out-dir");
            } else if (arg == "--lengths") {
                bench.lengths = parse_i32_list(need_value("--lengths"));
            } else if (arg == "--decode-tokens") {
                bench.decode_tokens = std::stoi(need_value("--decode-tokens"));
            } else if (arg == "--repeats") {
                bench.repeats = std::stoi(need_value("--repeats"));
            } else if (arg == "--n-cpu-moe-layers") {
                bench.n_cpu_moe_layers = std::stoi(need_value("--n-cpu-moe-layers"));
            } else if (arg == "--n-cpu-moe-sweep") {
                bench.n_cpu_moe_sweep = parse_i32_list(need_value("--n-cpu-moe-sweep"));
            } else if (arg == "--skip-model-bench") {
                bench.skip_model_bench = std::stoi(need_value("--skip-model-bench")) != 0;
            } else if (arg == "--skip-microbench") {
                bench.skip_microbench = std::stoi(need_value("--skip-microbench")) != 0;
            } else if (arg == "--profile-all-nodes") {
                bench.profile_all_nodes = std::stoi(need_value("--profile-all-nodes")) != 0;
            } else if (arg == "--expert-bytes") {
                if (!parse_size_bytes(need_value("--expert-bytes"), bench.expert_bytes)) {
                    throw std::invalid_argument("invalid --expert-bytes");
                }
            } else if (arg == "--micro-experts") {
                bench.micro_experts = std::stoi(need_value("--micro-experts"));
            } else if (arg == "--micro-d-model") {
                bench.micro_d_model = std::stoi(need_value("--micro-d-model"));
            } else if (arg == "--micro-d-ff") {
                bench.micro_d_ff = std::stoi(need_value("--micro-d-ff"));
            } else if (arg == "--micro-prefill-tokens") {
                bench.micro_prefill_tokens = std::stoi(need_value("--micro-prefill-tokens"));
            } else if (arg == "--micro-decode-tokens") {
                bench.micro_decode_tokens = std::stoi(need_value("--micro-decode-tokens"));
            } else {
                common_args.push_back(arg);
            }
        } catch (const std::exception & e) {
            LOG_ERR("%s: %s\n", __func__, e.what());
            return false;
        }
    }

    if (bench.lengths.empty() || bench.repeats <= 0 || bench.decode_tokens < 0) {
        LOG_ERR("%s: invalid benchmark dimensions\n", __func__);
        return false;
    }
    if (bench.mode != "cpu_full" && bench.mode != "moe_cpu_offload" && bench.mode != "moe_cpu_sweep") {
        LOG_ERR("%s: unknown mode '%s'\n", __func__, bench.mode.c_str());
        return false;
    }

    return true;
}

static std::vector<char *> argv_from_strings(std::vector<std::string> & args) {
    std::vector<char *> out;
    out.reserve(args.size());
    for (auto & arg : args) {
        out.push_back(arg.data());
    }
    return out;
}

static bool parse_layer_name(const char * cname, std::string & base, int32_t & layer) {
    const std::string name = cname ? cname : "";
    const size_t dash = name.rfind('-');
    if (dash == std::string::npos || dash + 1 >= name.size()) {
        base = name;
        layer = -1;
        return false;
    }

    for (size_t i = dash + 1; i < name.size(); ++i) {
        if (!std::isdigit((unsigned char) name[i])) {
            base = name;
            layer = -1;
            return false;
        }
    }

    base = name.substr(0, dash);
    layer = std::stoi(name.substr(dash + 1));
    return true;
}

static bool is_attention_node(const std::string & base) {
    return base == "fattn" ||
           base == "v_cont" ||
           base == "kq" ||
           base == "kqv" ||
           base == "kqv_out" ||
           base.rfind("kq_", 0) == 0 ||
           base.rfind("kqv_", 0) == 0 ||
           base.rfind("fattn_", 0) == 0;
}

static bool is_moe_node(const std::string & base) {
    return base.rfind("ffn_moe_", 0) == 0;
}

static bool eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    auto * cb = (callback_data *) user_data;
    std::string base;
    int32_t layer = -1;
    parse_layer_name(t->name, base, layer);
    const bool interesting = layer >= 0 && (is_attention_node(base) || is_moe_node(base));

    if (ask) {
        if (cb->profile_all_nodes || interesting) {
            cb->t_start_us = ggml_time_us();
            return true;
        }
        return false;
    }

    if (interesting && cb->records) {
        node_record rec;
        rec.mode = cb->mode;
        rec.phase = cb->phase;
        rec.name = base;
        rec.seq_len = cb->seq_len;
        rec.repeat = cb->repeat;
        rec.layer = layer;
        rec.time_us = ggml_time_us() - cb->t_start_us;
        cb->records->push_back(std::move(rec));
    }

    return true;
}

static void trim_tensor_overrides(std::vector<llama_model_tensor_buft_override> & overrides) {
    while (!overrides.empty() && overrides.back().pattern == nullptr) {
        overrides.pop_back();
    }
}

static void terminate_tensor_overrides(std::vector<llama_model_tensor_buft_override> & overrides) {
    if (overrides.empty() || overrides.back().pattern != nullptr) {
        overrides.push_back({ nullptr, nullptr });
    }
}

static void apply_cpu_moe_overrides(common_params & params, int32_t n_cpu_moe_layers, std::vector<std::string> & owned_patterns) {
    trim_tensor_overrides(params.tensor_buft_overrides);

    if (n_cpu_moe_layers < 0) {
        params.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
    } else {
        owned_patterns.clear();
        owned_patterns.reserve((size_t) n_cpu_moe_layers);
        for (int32_t il = 0; il < n_cpu_moe_layers; ++il) {
            owned_patterns.push_back(llm_ffn_exps_block_regex(il));
            params.tensor_buft_overrides.push_back({ owned_patterns.back().c_str(), ggml_backend_cpu_buffer_type() });
        }
    }

    terminate_tensor_overrides(params.tensor_buft_overrides);
}

static common_params make_run_params(const common_params & base, const bench_params & bench, const std::string & mode, int32_t n_cpu_moe_layers, std::vector<std::string> & owned_patterns) {
    common_params params = base;

    if (mode == "cpu_full") {
        params.n_gpu_layers = 0;
        params.devices = { nullptr };
        params.no_kv_offload = true;
        trim_tensor_overrides(params.tensor_buft_overrides);
        terminate_tensor_overrides(params.tensor_buft_overrides);
        return params;
    }

    if (mode == "moe_cpu_offload" || mode == "moe_cpu_sweep") {
        params.no_kv_offload = true;
        apply_cpu_moe_overrides(params, n_cpu_moe_layers, owned_patterns);
        return params;
    }

    (void) bench;
    return params;
}

static int64_t meta_i64(const llama_model * model, const std::string & key, int64_t def) {
    char buf[128];
    const int32_t n = llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
    if (n < 0) {
        return def;
    }
    char * end = nullptr;
    const int64_t v = std::strtoll(buf, &end, 10);
    return end == buf ? def : v;
}

static std::string meta_str(const llama_model * model, const std::string & key, const std::string & def) {
    char buf[256];
    const int32_t n = llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
    if (n < 0) {
        return def;
    }
    return buf;
}

static std::vector<llama_token> make_tokens(const llama_vocab * vocab, int32_t n_tokens) {
    const int32_t n_vocab = std::max(2, llama_vocab_n_tokens(vocab));
    std::vector<llama_token> tokens((size_t) n_tokens);
    for (int32_t i = 0; i < n_tokens; ++i) {
        tokens[i] = 1 + (i % (n_vocab - 1));
    }
    return tokens;
}

static bool decode_range(llama_context * ctx, llama_batch & batch, const std::vector<llama_token> & tokens, int32_t pos0, int32_t n_tokens, bool logits_last) {
    const uint32_t n_batch = std::max<uint32_t>(1, llama_n_batch(ctx));
    int32_t done = 0;
    while (done < n_tokens) {
        const int32_t cur = std::min<int32_t>((int32_t) n_batch, n_tokens - done);
        common_batch_clear(batch);
        for (int32_t i = 0; i < cur; ++i) {
            const bool logits = logits_last && (done + i + 1 == n_tokens);
            common_batch_add(batch, tokens[(size_t) done + i], pos0 + done + i, { 0 }, logits);
        }
        if (llama_decode(ctx, batch) != 0) {
            return false;
        }
        done += cur;
    }
    llama_synchronize(ctx);
    return true;
}

static void append_attention_summary(
        const std::vector<node_record> & nodes,
        std::ofstream & jsonl) {
    struct key {
        std::string mode;
        std::string phase;
        int32_t seq_len;
        int32_t repeat;
        int32_t layer;

        bool operator<(const key & other) const {
            return std::tie(mode, phase, seq_len, repeat, layer) <
                   std::tie(other.mode, other.phase, other.seq_len, other.repeat, other.layer);
        }
    };

    struct stat {
        int64_t time_us = 0;
        int64_t nodes = 0;
    };

    std::map<key, stat> stats;
    for (const auto & rec : nodes) {
        if (!is_attention_node(rec.name)) {
            continue;
        }
        auto & st = stats[{ rec.mode, rec.phase, rec.seq_len, rec.repeat, rec.layer }];
        st.time_us += rec.time_us;
        st.nodes++;
    }

    for (const auto & it : stats) {
        jsonl << "{\"kind\":\"attention_layer\""
              << ",\"mode\":\"" << json_escape(it.first.mode) << "\""
              << ",\"phase\":\"" << json_escape(it.first.phase) << "\""
              << ",\"seq_len\":" << it.first.seq_len
              << ",\"repeat\":" << it.first.repeat
              << ",\"layer\":" << it.first.layer
              << ",\"nodes\":" << it.second.nodes
              << ",\"time_us\":" << it.second.time_us
              << ",\"avg_node_us\":" << (it.second.nodes ? (double) it.second.time_us / (double) it.second.nodes : 0.0)
              << "}\n";
    }
}

static void write_copy_jsonl(const std::vector<copy_record> & copies, std::ofstream & jsonl) {
    for (const auto & rec : copies) {
        jsonl << "{\"kind\":\"moe_copy_runtime\""
              << ",\"mode\":\"" << json_escape(rec.mode) << "\""
              << ",\"phase\":\"" << json_escape(rec.phase) << "\""
              << ",\"seq_len\":" << rec.seq_len
              << ",\"repeat\":" << rec.repeat
              << ",\"calls\":" << rec.calls
              << ",\"bytes\":" << rec.bytes
              << ",\"time_us\":" << rec.time_us
              << "}\n";
    }
}

static bool run_model_bench(
        const bench_params & bench,
        common_params params,
        const std::string & mode_label,
        std::vector<node_record> & node_records,
        std::vector<copy_record> & copy_records,
        model_info & info) {
    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (model == nullptr) {
        LOG_ERR("%s: failed to load model for mode %s\n", __func__, mode_label.c_str());
        return false;
    }

    const std::string arch = meta_str(model, "general.architecture", "");
    info.n_layer  = llama_model_n_layer(model);
    info.n_embd   = llama_model_n_embd(model);
    info.n_expert = (int32_t) meta_i64(model, arch + ".expert_count", info.n_expert);
    info.n_ff_exp = (int32_t) meta_i64(model, arch + ".expert_feed_forward_length", info.n_ff_exp);

    callback_data cb;
    cb.profile_all_nodes = bench.profile_all_nodes;
    cb.mode = mode_label;
    cb.records = &node_records;

    params.cb_eval = eval_callback;
    params.cb_eval_user_data = &cb;

    llama_context_params ctx_params = common_context_params_to_llama(params);
    ctx_params.n_seq_max = 1;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        LOG_ERR("%s: failed to create context for mode %s\n", __func__, mode_label.c_str());
        llama_model_free(model);
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto * mem = llama_get_memory(ctx);

    const int32_t max_len = *std::max_element(bench.lengths.begin(), bench.lengths.end());
    std::vector<llama_token> prompt_tokens = make_tokens(vocab, max_len);
    std::vector<llama_token> decode_tokens = make_tokens(vocab, bench.decode_tokens);
    llama_batch batch = llama_batch_init((int32_t) std::max<uint32_t>(1, llama_n_batch(ctx)), 0, 1);

    ggml_backend_moe_copy_stats_set_enabled(true);

    for (int32_t seq_len : bench.lengths) {
        if (seq_len + bench.decode_tokens > (int32_t) llama_n_ctx(ctx)) {
            LOG_WRN("%s: skipping seq_len=%d because n_ctx=%u\n", __func__, seq_len, llama_n_ctx(ctx));
            continue;
        }

        for (int32_t rep = 0; rep < bench.repeats; ++rep) {
            llama_memory_clear(mem, false);

            cb.phase = "prefill";
            cb.seq_len = seq_len;
            cb.repeat = rep;
            ggml_backend_moe_copy_stats_reset();
            LOG_INF("bench: mode=%s seq_len=%d repeat=%d/%d phase=prefill start\n",
                    mode_label.c_str(), seq_len, rep + 1, bench.repeats);
            const int64_t t_prefill_start = ggml_time_us();
            if (!decode_range(ctx, batch, prompt_tokens, 0, seq_len, false)) {
                LOG_ERR("%s: prefill failed for mode=%s seq_len=%d repeat=%d\n", __func__, mode_label.c_str(), seq_len, rep);
                llama_batch_free(batch);
                llama_free(ctx);
                llama_model_free(model);
                return false;
            }
            const int64_t t_prefill_end = ggml_time_us();
            auto st_prefill = ggml_backend_moe_copy_stats_get();
            copy_records.push_back({ mode_label, "prefill", seq_len, rep, st_prefill.calls, st_prefill.bytes, st_prefill.time_us });
            LOG_INF("bench: mode=%s seq_len=%d repeat=%d/%d phase=prefill done total_ms=%.3f moe_copy_ms=%.3f\n",
                    mode_label.c_str(), seq_len, rep + 1, bench.repeats,
                    (t_prefill_end - t_prefill_start)/1000.0,
                    st_prefill.time_us/1000.0);

            cb.phase = "decode";
            cb.seq_len = seq_len;
            cb.repeat = rep;
            ggml_backend_moe_copy_stats_reset();
            LOG_INF("bench: mode=%s seq_len=%d repeat=%d/%d phase=decode start tokens=%d\n",
                    mode_label.c_str(), seq_len, rep + 1, bench.repeats, bench.decode_tokens);
            const int64_t t_decode_start = ggml_time_us();
            for (int32_t i = 0; i < bench.decode_tokens; ++i) {
                std::vector<llama_token> one = { decode_tokens[(size_t) i] };
                if (!decode_range(ctx, batch, one, seq_len + i, 1, true)) {
                    LOG_ERR("%s: decode failed for mode=%s seq_len=%d repeat=%d token=%d\n", __func__, mode_label.c_str(), seq_len, rep, i);
                    llama_batch_free(batch);
                    llama_free(ctx);
                    llama_model_free(model);
                    return false;
                }
            }
            const int64_t t_decode_end = ggml_time_us();
            auto st_decode = ggml_backend_moe_copy_stats_get();
            copy_records.push_back({ mode_label, "decode", seq_len, rep, st_decode.calls, st_decode.bytes, st_decode.time_us });
            LOG_INF("bench: mode=%s seq_len=%d repeat=%d/%d phase=decode done total_ms=%.3f moe_copy_ms=%.3f\n",
                    mode_label.c_str(), seq_len, rep + 1, bench.repeats,
                    (t_decode_end - t_decode_start)/1000.0,
                    st_decode.time_us/1000.0);
        }
    }

    ggml_backend_moe_copy_stats_set_enabled(false);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return true;
}

static ggml_backend_dev_t first_gpu_device() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ||
            ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            return dev;
        }
    }
    return nullptr;
}

static std::vector<ggml_fp16_t> make_f16_data(size_t n) {
    std::vector<ggml_fp16_t> data(n);
    for (size_t i = 0; i < n; ++i) {
        const float v = ((int) (i % 17) - 8) / 64.0f;
        data[i] = ggml_fp32_to_fp16(v);
    }
    return data;
}

static bool run_h2d_microbench(const bench_params & bench, const model_info & info, std::vector<micro_record> & out) {
    ggml_backend_dev_t dev = first_gpu_device();
    if (dev == nullptr) {
        LOG_WRN("%s: no GPU/ACCEL backend found; skipping H2D microbench\n", __func__);
        return true;
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    if (host_buft == nullptr) {
        LOG_WRN("%s: device has no pinned host buffer type; skipping H2D microbench\n", __func__);
        return true;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        LOG_WRN("%s: failed to initialize backend; skipping H2D microbench\n", __func__);
        return true;
    }

    size_t bytes = bench.expert_bytes;
    if (bytes == 0) {
        const int32_t d_model = info.n_embd > 0 ? info.n_embd : bench.micro_d_model;
        const int32_t d_ff = info.n_ff_exp > 0 ? info.n_ff_exp : bench.micro_d_ff;
        bytes = (size_t) (2*d_model*d_ff + d_ff*d_model) * sizeof(ggml_fp16_t);
    }
    bytes = std::max<size_t>(bytes, 1);

    const size_t n_elems = (bytes + sizeof(int32_t) - 1) / sizeof(int32_t);
    ggml_init_params init_params = {
        /* .mem_size   = */ 2*ggml_tensor_overhead() + 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx_src = ggml_init(init_params);
    ggml_context * ctx_dst = ggml_init(init_params);
    ggml_tensor * src = ggml_new_tensor_1d(ctx_src, GGML_TYPE_I32, (int64_t) n_elems);
    ggml_tensor * dst = ggml_new_tensor_1d(ctx_dst, GGML_TYPE_I32, (int64_t) n_elems);

    ggml_backend_buffer_t host_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_src, host_buft);
    if (host_buf == nullptr) {
        ggml_free(ctx_dst);
        ggml_free(ctx_src);
        ggml_backend_free(backend);
        LOG_WRN("%s: failed to allocate pinned host buffer; skipping H2D microbench\n", __func__);
        return true;
    }
    ggml_backend_buffer_t gpu_buf = ggml_backend_alloc_ctx_tensors(ctx_dst, backend);
    if (gpu_buf == nullptr) {
        ggml_backend_buffer_free(host_buf);
        ggml_free(ctx_dst);
        ggml_free(ctx_src);
        ggml_backend_free(backend);
        LOG_WRN("%s: failed to allocate GPU buffer; skipping H2D microbench\n", __func__);
        return true;
    }
    ggml_backend_buffer_clear(host_buf, 0x7f);

    for (int32_t rep = 0; rep < bench.repeats; ++rep) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(dst));
        ggml_backend_synchronize(backend);
        const int64_t t1 = ggml_time_us();
        out.push_back({ "h2d_pinned", "copy", 0, rep, t1 - t0, (uint64_t) ggml_nbytes(dst) });
    }

    ggml_backend_buffer_free(gpu_buf);
    ggml_backend_buffer_free(host_buf);
    ggml_free(ctx_dst);
    ggml_free(ctx_src);
    ggml_backend_free(backend);
    return true;
}

static ggml_cgraph * build_group_graph(ggml_context * ctx, int32_t d_model, int32_t d_ff, int32_t n_experts, int32_t n_tokens) {
    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_model, d_ff, n_experts);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_model, 1, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_experts, n_tokens);
    ggml_tensor * y = ggml_mul_mat_id(ctx, w, x, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    return gf;
}

static ggml_cgraph * build_serial_graph(ggml_context * ctx, int32_t d_model, int32_t d_ff, int32_t n_experts, int32_t n_tokens) {
    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_model, d_ff, n_experts);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, n_tokens);
    ggml_tensor * acc = nullptr;
    for (int32_t i = 0; i < n_experts; ++i) {
        ggml_tensor * wi = ggml_view_2d(ctx, w, d_model, d_ff, w->nb[1], i*w->nb[2]);
        ggml_tensor * yi = ggml_mul_mat(ctx, wi, x);
        acc = acc ? ggml_add(ctx, acc, yi) : yi;
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, acc);
    return gf;
}

static bool fill_graph_inputs(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->data == nullptr || t->op != GGML_OP_NONE) {
            continue;
        }
        if (t->type == GGML_TYPE_F16) {
            auto data = make_f16_data((size_t) ggml_nelements(t));
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        } else if (t->type == GGML_TYPE_F32) {
            std::vector<float> data((size_t) ggml_nelements(t));
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = ((int) (i % 17) - 8) / 64.0f;
            }
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        } else if (t->type == GGML_TYPE_I32) {
            std::vector<int32_t> ids((size_t) ggml_nelements(t));
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    ids[(size_t) (i1*t->ne[0] + i0)] = (int32_t) i0;
                }
            }
            ggml_backend_tensor_set(t, ids.data(), 0, ggml_nbytes(t));
        }
    }
    return true;
}

static bool run_one_gemm_micro(
        ggml_backend_t backend,
        bool group,
        const std::string & phase,
        int32_t tokens,
        const bench_params & bench,
        const model_info & info,
        std::vector<micro_record> & out) {
    const int32_t d_model = info.n_embd > 0 ? info.n_embd : bench.micro_d_model;
    const int32_t d_ff = info.n_ff_exp > 0 ? info.n_ff_exp : bench.micro_d_ff;
    const int32_t n_experts = bench.micro_experts;

    const size_t mem_size = 64ULL*1024ULL*1024ULL;
    ggml_init_params init_params = {
        /* .mem_size   = */ mem_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(init_params);
    ggml_cgraph * gf = group ?
        build_group_graph(ctx, d_model, d_ff, n_experts, tokens) :
        build_serial_graph(ctx, d_model, d_ff, n_experts, tokens);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        LOG_WRN("%s: failed to allocate microbench tensors for %s/%s\n", __func__, group ? "group" : "serial", phase.c_str());
        ggml_free(ctx);
        return true;
    }

    fill_graph_inputs(ctx);

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        LOG_WRN("%s: warmup failed for %s/%s\n", __func__, group ? "group" : "serial", phase.c_str());
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return true;
    }

    for (int32_t rep = 0; rep < bench.repeats; ++rep) {
        const int64_t t0 = ggml_time_us();
        const enum ggml_status status = ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        const int64_t t1 = ggml_time_us();
        if (status != GGML_STATUS_SUCCESS) {
            LOG_WRN("%s: microbench compute failed for %s/%s\n", __func__, group ? "group" : "serial", phase.c_str());
            break;
        }
        out.push_back({ group ? "moe_group_gemm" : "moe_serial_gemm", phase, tokens, rep, t1 - t0, 0 });
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static bool run_gemm_microbench(const bench_params & bench, const model_info & info, std::vector<micro_record> & out) {
    ggml_backend_dev_t dev = first_gpu_device();
    if (dev == nullptr) {
        LOG_WRN("%s: no GPU/ACCEL backend found; skipping GEMM microbench\n", __func__);
        return true;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        LOG_WRN("%s: failed to initialize backend; skipping GEMM microbench\n", __func__);
        return true;
    }

    run_one_gemm_micro(backend, true,  "prefill", bench.micro_prefill_tokens, bench, info, out);
    run_one_gemm_micro(backend, false, "prefill", bench.micro_prefill_tokens, bench, info, out);
    run_one_gemm_micro(backend, true,  "decode",  bench.micro_decode_tokens,  bench, info, out);
    run_one_gemm_micro(backend, false, "decode",  bench.micro_decode_tokens,  bench, info, out);

    ggml_backend_free(backend);
    return true;
}

static void write_raw_jsonl(
        const std::filesystem::path & path,
        const std::vector<node_record> & nodes,
        const std::vector<copy_record> & copies,
        const std::vector<micro_record> & micros) {
    std::ofstream jsonl(path);
    for (const auto & rec : nodes) {
        jsonl << "{\"kind\":\"node\""
              << ",\"mode\":\"" << json_escape(rec.mode) << "\""
              << ",\"phase\":\"" << json_escape(rec.phase) << "\""
              << ",\"seq_len\":" << rec.seq_len
              << ",\"repeat\":" << rec.repeat
              << ",\"layer\":" << rec.layer
              << ",\"name\":\"" << json_escape(rec.name) << "\""
              << ",\"time_us\":" << rec.time_us
              << "}\n";
    }
    append_attention_summary(nodes, jsonl);
    write_copy_jsonl(copies, jsonl);
    for (const auto & rec : micros) {
        jsonl << "{\"kind\":\"" << json_escape(rec.kind) << "\""
              << ",\"phase\":\"" << json_escape(rec.phase) << "\""
              << ",\"tokens\":" << rec.tokens
              << ",\"repeat\":" << rec.repeat
              << ",\"time_us\":" << rec.time_us
              << ",\"bytes\":" << rec.bytes
              << "}\n";
    }
}

static void write_summary_csv(
        const std::filesystem::path & path,
        const std::vector<node_record> & nodes,
        const std::vector<copy_record> & copies,
        const std::vector<micro_record> & micros) {
    struct bucket {
        std::vector<double> values;
    };

    std::map<std::string, bucket> buckets;

    struct attn_key {
        std::string mode;
        std::string phase;
        int32_t seq_len;
        int32_t repeat;
        int32_t layer;
        bool operator<(const attn_key & o) const {
            return std::tie(mode, phase, seq_len, repeat, layer) < std::tie(o.mode, o.phase, o.seq_len, o.repeat, o.layer);
        }
    };
    std::map<attn_key, int64_t> attn;
    for (const auto & rec : nodes) {
        if (is_attention_node(rec.name)) {
            attn[{ rec.mode, rec.phase, rec.seq_len, rec.repeat, rec.layer }] += rec.time_us;
        }
    }
    for (const auto & it : attn) {
        std::ostringstream key;
        key << "attention_layer," << it.first.mode << "," << it.first.phase << "," << it.first.seq_len << "," << it.first.layer << ",time_us";
        buckets[key.str()].values.push_back((double) it.second);
    }

    for (const auto & rec : copies) {
        std::ostringstream kt;
        kt << "moe_copy_runtime," << rec.mode << "," << rec.phase << "," << rec.seq_len << ",-1,time_us";
        buckets[kt.str()].values.push_back((double) rec.time_us);

        std::ostringstream kb;
        kb << "moe_copy_runtime," << rec.mode << "," << rec.phase << "," << rec.seq_len << ",-1,bytes";
        buckets[kb.str()].values.push_back((double) rec.bytes);
    }

    for (const auto & rec : micros) {
        std::ostringstream key;
        key << rec.kind << ",micro," << rec.phase << "," << rec.tokens << ",-1,time_us";
        buckets[key.str()].values.push_back((double) rec.time_us);
        if (rec.bytes > 0) {
            std::ostringstream kb;
            kb << rec.kind << ",micro," << rec.phase << "," << rec.tokens << ",-1,bytes";
            buckets[kb.str()].values.push_back((double) rec.bytes);
        }
    }

    std::ofstream csv(path);
    csv << "kind,mode,phase,seq_len_or_tokens,layer,metric,n,avg,stddev,min,max\n";
    for (const auto & it : buckets) {
        const auto & values = it.second.values;
        const double avg = std::accumulate(values.begin(), values.end(), 0.0) / (double) values.size();
        double var = 0.0;
        for (double v : values) {
            var += (v - avg) * (v - avg);
        }
        const double stddev = std::sqrt(var / (double) values.size());
        const auto minmax = std::minmax_element(values.begin(), values.end());
        csv << it.first << "," << values.size() << "," << avg << "," << stddev << "," << *minmax.first << "," << *minmax.second << "\n";
    }
}

static void write_report(const std::filesystem::path & path, const bench_params & bench, const model_info & info, int argc, char ** argv) {
    std::ofstream report(path);
    report << "# Qwen3-VL MoE Benchmark Report\n\n";
    report << "## Command\n\n```text\n";
    for (int i = 0; i < argc; ++i) {
        if (i) {
            report << " ";
        }
        report << argv[i];
    }
    report << "\n```\n\n";
    report << "## Config\n\n";
    report << "- mode: `" << bench.mode << "`\n";
    report << "- repeats: `" << bench.repeats << "`\n";
    report << "- decode_tokens: `" << bench.decode_tokens << "`\n";
    report << "- profile_all_nodes: `" << (bench.profile_all_nodes ? 1 : 0) << "`\n";
    report << "- model layers: `" << info.n_layer << "`\n";
    report << "- model embedding: `" << info.n_embd << "`\n";
    report << "- experts: `" << info.n_expert << "`\n";
    report << "- expert FFN: `" << info.n_ff_exp << "`\n\n";
    report << "## Outputs\n\n";
    report << "- `results.jsonl`: raw node, attention-layer, runtime MoE-copy, and microbench records\n";
    report << "- `summary.csv`: aggregated averages and stddevs\n";
    report << "- `plots/*.svg`: generated by `plot_qwen3vl_moe_bench.py`\n\n";
    report << "Note: runtime MoE-copy profiling synchronizes the destination backend around used-expert copies, so it is for measurement rather than maximum-throughput serving.\n";
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    bench_params bench;
    std::vector<std::string> common_arg_strings;
    if (!parse_custom_args(argc, argv, bench, common_arg_strings)) {
        print_usage(argv[0]);
        return 1;
    }

    common_params params;
    common_init();

    auto common_argv = argv_from_strings(common_arg_strings);
    if (!common_params_parse((int) common_argv.size(), common_argv.data(), params, LLAMA_EXAMPLE_BENCH, [](int, char ** argv_inner) {
        print_usage(argv_inner[0]);
    })) {
        return 1;
    }

    const int32_t max_len = *std::max_element(bench.lengths.begin(), bench.lengths.end());
    const int32_t n_ctx_req = max_len + bench.decode_tokens;
    if (params.n_ctx < n_ctx_req) {
        params.n_ctx = n_ctx_req;
    }
    params.no_perf = true;
    params.warmup = false;

    std::filesystem::create_directories(bench.out_dir);

    llama_backend_init();
    llama_numa_init(params.numa);

    std::vector<node_record> node_records;
    std::vector<copy_record> copy_records;
    std::vector<micro_record> micro_records;
    model_info info;
    bool ok = true;

    if (!bench.skip_model_bench) {
        if (bench.mode == "moe_cpu_sweep") {
            for (int32_t n_cpu_moe : bench.n_cpu_moe_sweep) {
                std::vector<std::string> owned_patterns;
                std::string mode_label = "moe_cpu_sweep_n" + std::to_string(n_cpu_moe);
                common_params run_params = make_run_params(params, bench, "moe_cpu_sweep", n_cpu_moe, owned_patterns);
                ok = run_model_bench(bench, run_params, mode_label, node_records, copy_records, info) && ok;
            }
        } else {
            std::vector<std::string> owned_patterns;
            common_params run_params = make_run_params(params, bench, bench.mode, bench.n_cpu_moe_layers, owned_patterns);
            ok = run_model_bench(bench, run_params, bench.mode, node_records, copy_records, info) && ok;
        }
    }

    if (!bench.skip_microbench) {
        ok = run_h2d_microbench(bench, info, micro_records) && ok;
        ok = run_gemm_microbench(bench, info, micro_records) && ok;
    }

    const std::filesystem::path out_dir = bench.out_dir;
    write_raw_jsonl(out_dir / "results.jsonl", node_records, copy_records, micro_records);
    write_summary_csv(out_dir / "summary.csv", node_records, copy_records, micro_records);
    write_report(out_dir / "report.md", bench, info, argc, argv);

    llama_backend_free();

    LOG_INF("wrote %s, %s, %s\n",
            (out_dir / "results.jsonl").string().c_str(),
            (out_dir / "summary.csv").string().c_str(),
            (out_dir / "report.md").string().c_str());

    return ok ? 0 : 1;
}
