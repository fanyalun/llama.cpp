#include "arg.h"
#include "common.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <climits>
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
#include <utility>
#include <vector>

struct bench_params {
    std::string mode = "moe_cpu_offload";
    std::string attention_backend = "gpu";
    std::string attention_placement = "kv_gpu_attn_gpu";
    std::string out_dir = "qwen3vl_moe_bench_results";
    std::string prompt_json;
    std::vector<int32_t> lengths = { 1024, 4096, 8192, 16384 };
    std::vector<int32_t> micro_tokens = { 1024, 2048, 4096, 8192, 16384, 32768 };
    std::vector<int32_t> micro_experts_sweep = { 1, 2, 3, 4, 5, 6, 7, 8 };
    std::vector<int32_t> micro_alpha_pcts = { 0, 25, 50, 75, 100 };
    std::vector<int32_t> n_cpu_moe_sweep = { 48, 36, 24, 12, 0 };
    int32_t decode_tokens = 16;
    int32_t repeats = 3;
    int32_t n_cpu_moe_layers = -1;
    bool skip_model_bench = false;
    bool skip_microbench = false;
    bool skip_gemm_microbench = false;
    bool profile_all_nodes = true;
    int32_t micro_experts = 128;
    int32_t micro_d_model = 2048;
    int32_t micro_d_ff = 4096;
    int32_t micro_prefill_tokens = 512;
    int32_t micro_decode_tokens = 1;
    int32_t route_layer = 10;
    int32_t route_max_prompts = 16;
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
    int32_t active_tokens = 0;
    int32_t experts = 0;
    int32_t alpha_pct = -1;
    int32_t group_experts = 0;
    int32_t serial_experts = 0;
    int32_t active_experts = 0;
    int32_t sample = -1;
    int32_t repeat = 0;
    int64_t time_us = 0;
    uint64_t bytes = 0;
};

struct route_record {
    int32_t seq_len = 0;
    int32_t sample = 0;
    int32_t original_tokens = 0;
    int32_t layer = -1;
    std::vector<int64_t> expert_counts;
};

struct callback_data {
    bool profile_all_nodes = true;
    bool collect_routes = false;
    std::string mode;
    std::string phase;
    int32_t seq_len = 0;
    int32_t repeat = 0;
    int32_t route_layer = -1;
    int64_t t_start_us = 0;
    std::vector<node_record> * records = nullptr;
    std::vector<int64_t> * route_counts = nullptr;
    std::vector<uint8_t> route_data;
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
    LOG("  --attention-placement <kv_gpu_attn_gpu|kv_cpu_attn_cpu|kv_cpu_attn_gpu|all>\n");
    LOG("                                                    default: kv_gpu_attn_gpu\n");
    LOG("  --attention-backend <cpu|gpu>                     compatibility alias for kv_cpu_attn_cpu / kv_gpu_attn_gpu\n");
    LOG("  --out-dir <dir>                                  default: qwen3vl_moe_bench_results\n");
    LOG("  --prompt-json <path>                             collect real layer routes from AIME/OpenAI allresults JSON\n");
    LOG("  --route-layer <n>                                0-based layer id for token2expert collection; default: 10\n");
    LOG("  --route-max-prompts <n>                          max prompts per length for route collection; 0 means all; default: 16\n");
    LOG("  --lengths <csv>                                  default: 1024,4096,8192,16384\n");
    LOG("  --micro-tokens <csv>                             default: 1024,2048,4096,8192,16384,32768\n");
    LOG("  --micro-alpha-pcts <csv>                         default: 0,25,50,75,100\n");
    LOG("  --micro-experts-sweep <csv>                      compatibility option; alpha sweep ignores it\n");
    LOG("  --decode-tokens <n>                              default: 16\n");
    LOG("  --repeats <n>                                    default: 3\n");
    LOG("  --n-cpu-moe-layers <n>                           for moe_cpu_offload; -1 means all experts\n");
    LOG("  --n-cpu-moe-sweep <csv>                          default: 48,36,24,12,0\n");
    LOG("  --skip-model-bench <0|1>                         default: 0\n");
    LOG("  --skip-microbench <0|1>                          default: 0\n");
    LOG("  --skip-gemm-microbench <0|1>                     default: 0; still runs expert H2D copy\n");
    LOG("  --profile-all-nodes <0|1>                        default: 1, exact but slower node timing\n");
    LOG("  --expert-bytes <bytes|MiB|GiB>                   override reported expert copy bytes\n");
    LOG("  --micro-experts <n>                              fallback active prefill experts when metadata is absent; default: 128\n");
    LOG("  --micro-d-model <n>                              default: model n_embd or 2048\n");
    LOG("  --micro-d-ff <n>                                 default: model expert FFN or 4096\n");
    LOG("  --micro-prefill-tokens <n>                       compatibility alias for --micro-tokens <n>\n");
    LOG("  --micro-decode-tokens <n>                        default: 1 active token for decode labels\n\n");
    LOG("recommended attention placement run:\n");
    LOG("  %s -m model.gguf --mode moe_cpu_offload --attention-placement all --flash-attn on -ngl auto --lengths 1024,4096,8192,16384 --decode-tokens 16\n\n", argv0);
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
        auto optional_bool_value = [&](bool default_value) -> bool {
            if (i + 1 >= argc || std::string(argv[i + 1]).rfind("-", 0) == 0) {
                return default_value;
            }
            return std::stoi(argv[++i]) != 0;
        };

        try {
            if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                common_args.push_back(arg);
            } else if (arg == "--mode") {
                bench.mode = need_value("--mode");
            } else if (arg == "--attention-backend") {
                bench.attention_backend = need_value("--attention-backend");
                bench.attention_placement = bench.attention_backend == "cpu" ? "kv_cpu_attn_cpu" : "kv_gpu_attn_gpu";
            } else if (arg == "--attention-placement") {
                bench.attention_placement = need_value("--attention-placement");
            } else if (arg == "--out-dir") {
                bench.out_dir = need_value("--out-dir");
            } else if (arg == "--prompt-json") {
                bench.prompt_json = need_value("--prompt-json");
            } else if (arg == "--route-layer") {
                bench.route_layer = std::stoi(need_value("--route-layer"));
            } else if (arg == "--route-max-prompts") {
                bench.route_max_prompts = std::stoi(need_value("--route-max-prompts"));
            } else if (arg == "--lengths") {
                bench.lengths = parse_i32_list(need_value("--lengths"));
            } else if (arg == "--micro-tokens") {
                bench.micro_tokens = parse_i32_list(need_value("--micro-tokens"));
            } else if (arg == "--micro-experts-sweep") {
                bench.micro_experts_sweep = parse_i32_list(need_value("--micro-experts-sweep"));
            } else if (arg == "--micro-alpha-pcts") {
                bench.micro_alpha_pcts = parse_i32_list(need_value("--micro-alpha-pcts"));
            } else if (arg == "--decode-tokens") {
                bench.decode_tokens = std::stoi(need_value("--decode-tokens"));
            } else if (arg == "--repeats") {
                bench.repeats = std::stoi(need_value("--repeats"));
            } else if (arg == "--n-cpu-moe-layers") {
                bench.n_cpu_moe_layers = std::stoi(need_value("--n-cpu-moe-layers"));
            } else if (arg == "--n-cpu-moe-sweep") {
                bench.n_cpu_moe_sweep = parse_i32_list(need_value("--n-cpu-moe-sweep"));
            } else if (arg == "--skip-model-bench") {
                bench.skip_model_bench = optional_bool_value(true);
            } else if (arg == "--skip-microbench") {
                bench.skip_microbench = optional_bool_value(true);
            } else if (arg == "--skip-gemm-microbench") {
                bench.skip_gemm_microbench = optional_bool_value(true);
            } else if (arg == "--profile-all-nodes") {
                bench.profile_all_nodes = optional_bool_value(true);
            } else if (arg == "--expert-bytes") {
                if (!parse_size_bytes(need_value("--expert-bytes"), bench.expert_bytes)) {
                    throw std::invalid_argument("invalid --expert-bytes");
                }
            } else if (arg == "--micro-experts") {
                bench.micro_experts = std::stoi(need_value("--micro-experts"));
                bench.micro_experts_sweep = { bench.micro_experts };
            } else if (arg == "--micro-d-model") {
                bench.micro_d_model = std::stoi(need_value("--micro-d-model"));
            } else if (arg == "--micro-d-ff") {
                bench.micro_d_ff = std::stoi(need_value("--micro-d-ff"));
            } else if (arg == "--micro-prefill-tokens") {
                bench.micro_prefill_tokens = std::stoi(need_value("--micro-prefill-tokens"));
                bench.micro_tokens = { bench.micro_prefill_tokens };
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

    if (bench.lengths.empty() || bench.micro_tokens.empty() || bench.micro_alpha_pcts.empty() || bench.repeats <= 0 || bench.decode_tokens < 0 ||
            bench.route_layer < 0 || bench.route_max_prompts < 0) {
        LOG_ERR("%s: invalid benchmark dimensions\n", __func__);
        return false;
    }
    if (bench.mode != "cpu_full" && bench.mode != "moe_cpu_offload" && bench.mode != "moe_cpu_sweep") {
        LOG_ERR("%s: unknown mode '%s'\n", __func__, bench.mode.c_str());
        return false;
    }
    if (bench.attention_backend != "cpu" && bench.attention_backend != "gpu") {
        LOG_ERR("%s: unknown attention backend '%s'\n", __func__, bench.attention_backend.c_str());
        return false;
    }
    if (bench.attention_placement != "kv_gpu_attn_gpu" &&
            bench.attention_placement != "kv_cpu_attn_cpu" &&
            bench.attention_placement != "kv_cpu_attn_gpu" &&
            bench.attention_placement != "all") {
        LOG_ERR("%s: unknown attention placement '%s'\n", __func__, bench.attention_placement.c_str());
        return false;
    }
    for (const int32_t h : bench.micro_experts_sweep) {
        if (h <= 0) {
            LOG_ERR("%s: invalid micro expert count %d\n", __func__, h);
            return false;
        }
    }
    for (const int32_t alpha_pct : bench.micro_alpha_pcts) {
        if (alpha_pct < 0 || alpha_pct > 100) {
            LOG_ERR("%s: invalid micro alpha percent %d\n", __func__, alpha_pct);
            return false;
        }
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
    return base == "__fattn__" ||
           base == "attention_kv_h2d" ||
           base == "fattn_mla" ||
           base == "v_cont" ||
           base == "kq" ||
           base == "kqv" ||
           base == "kqv_mla" ||
           base == "kq_scaled" ||
           base.rfind("kq_", 0) == 0;
}

static bool is_moe_node(const std::string & base) {
    return base.rfind("ffn_moe_", 0) == 0;
}

static bool is_route_node(const std::string & base) {
    return base == "ffn_moe_topk";
}

static bool eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    auto * cb = (callback_data *) user_data;
    std::string base;
    int32_t layer = -1;
    parse_layer_name(t->name, base, layer);
    const bool interesting = layer >= 0 && (is_attention_node(base) || is_moe_node(base));
    const bool route_target = cb->collect_routes &&
                              cb->phase == "prefill" &&
                              cb->route_counts != nullptr &&
                              layer == cb->route_layer &&
                              is_route_node(base);

    if (ask) {
        if (route_target || cb->profile_all_nodes || interesting) {
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

    if (route_target && t->type == GGML_TYPE_I32) {
        const size_t n_bytes = ggml_nbytes(t);
        const int32_t * ids = nullptr;
        if (ggml_backend_buffer_is_host(t->buffer)) {
            ids = (const int32_t *) t->data;
        } else {
            cb->route_data.resize(n_bytes);
            ggml_backend_tensor_get(t, cb->route_data.data(), 0, n_bytes);
            ids = (const int32_t *) cb->route_data.data();
        }

        const size_t n_ids = n_bytes / sizeof(int32_t);
        for (size_t i = 0; i < n_ids; ++i) {
            const int32_t expert = ids[i];
            if (expert >= 0) {
                if ((size_t) expert >= cb->route_counts->size()) {
                    cb->route_counts->resize((size_t) expert + 1, 0);
                }
                (*cb->route_counts)[(size_t) expert]++;
            }
        }
    }

    return true;
}

static uint64_t append_attention_kv_copy_records(
        const std::string & mode,
        const std::string & phase,
        int32_t seq_len,
        int32_t repeat,
        std::vector<node_record> & node_records) {
    uint64_t total_us = 0;
    const size_t n = ggml_backend_kv_copy_stats_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_kv_copy_record kv = {};
        if (!ggml_backend_kv_copy_stats_get(i, &kv)) {
            continue;
        }

        node_record rec;
        rec.mode = mode;
        rec.phase = phase;
        rec.name = "attention_kv_h2d";
        rec.seq_len = seq_len;
        rec.repeat = repeat;
        rec.layer = kv.layer;
        rec.time_us = (int64_t) kv.time_us;
        node_records.push_back(std::move(rec));
        total_us += kv.time_us;
    }
    return total_us;
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

static bool placement_kv_gpu(const std::string & placement) {
    return placement == "kv_gpu_attn_gpu";
}

static bool placement_attn_gpu(const std::string & placement) {
    return placement == "kv_gpu_attn_gpu" || placement == "kv_cpu_attn_gpu";
}

static std::vector<std::string> attention_placements_to_run(const bench_params & bench) {
    if (bench.mode == "cpu_full") {
        return { "kv_cpu_attn_cpu" };
    }
    if (bench.attention_placement == "all") {
        return { "kv_gpu_attn_gpu", "kv_cpu_attn_cpu", "kv_cpu_attn_gpu" };
    }
    return { bench.attention_placement };
}

static bool any_placement_attn_gpu(const bench_params & bench) {
    for (const auto & placement : attention_placements_to_run(bench)) {
        if (placement_attn_gpu(placement)) {
            return true;
        }
    }
    return false;
}

static common_params make_run_params(const common_params & base, const bench_params & bench, const std::string & mode, const std::string & attention_placement, int32_t n_cpu_moe_layers, std::vector<std::string> & owned_patterns) {
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
        params.no_kv_offload = !placement_kv_gpu(attention_placement);
        apply_cpu_moe_overrides(params, n_cpu_moe_layers, owned_patterns);
        return params;
    }

    (void) bench;
    return params;
}

static std::string make_mode_label(const std::string & attention_placement, const std::string & mode, int32_t n_cpu_moe_layers) {
    if (mode == "moe_cpu_sweep") {
        return "moe_cpu_sweep_n" + std::to_string(n_cpu_moe_layers) + "_" + attention_placement;
    }
    if (mode == "moe_cpu_offload") {
        return "moe_cpu_offload_" + attention_placement;
    }
    return mode;
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

static std::string json_string_unescape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 >= s.size()) {
            out.push_back(s[i]);
            continue;
        }
        const char c = s[++i];
        switch (c) {
            case 'n':  out.push_back('\n'); break;
            case 'r':  out.push_back('\r'); break;
            case 't':  out.push_back('\t'); break;
            case '"':  out.push_back('"');  break;
            case '\\': out.push_back('\\'); break;
            case '/':  out.push_back('/');  break;
            default:
                out.push_back('\\');
                out.push_back(c);
                break;
        }
    }
    return out;
}

static void replace_all(std::string & s, const std::string & from, const std::string & to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

static std::string html_unescape_minimal(std::string s) {
    replace_all(s, "&gt;", ">");
    replace_all(s, "&lt;", "<");
    replace_all(s, "&amp;", "&");
    replace_all(s, "&quot;", "\"");
    replace_all(s, "&#39;", "'");
    return s;
}

static std::vector<std::string> load_aime_prompt_json(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        LOG_ERR("%s: failed to open prompt JSON '%s'\n", __func__, path.c_str());
        return {};
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<std::string> prompts;

    size_t pos = 0;
    while ((pos = text.find("<h3>Prompt conversation</h3>", pos)) != std::string::npos) {
        const size_t pre0 = text.find("<pre>", pos);
        if (pre0 == std::string::npos) {
            break;
        }
        const size_t content0 = pre0 + std::strlen("<pre>");
        const size_t pre1 = text.find("</pre>", content0);
        if (pre1 == std::string::npos) {
            break;
        }

        std::string prompt = html_unescape_minimal(json_string_unescape(text.substr(content0, pre1 - content0)));
        while (!prompt.empty() && std::isspace((unsigned char) prompt.front())) {
            prompt.erase(prompt.begin());
        }
        while (!prompt.empty() && std::isspace((unsigned char) prompt.back())) {
            prompt.pop_back();
        }
        if (!prompt.empty()) {
            prompts.push_back(std::move(prompt));
        }
        pos = pre1 + std::strlen("</pre>");
    }

    LOG_INF("%s: loaded %zu prompt(s) from %s\n", __func__, prompts.size(), path.c_str());
    return prompts;
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
        const std::string & attention_placement,
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
    ctx_params.offload_attn = placement_attn_gpu(attention_placement) ? 1 : 0;
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
    ggml_backend_kv_copy_stats_set_enabled(attention_placement == "kv_cpu_attn_gpu");

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
            ggml_backend_kv_copy_stats_reset();
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
            const uint64_t kv_prefill_us = append_attention_kv_copy_records(mode_label, "prefill", seq_len, rep, node_records);
            copy_records.push_back({ mode_label, "prefill", seq_len, rep, st_prefill.calls, st_prefill.bytes, st_prefill.time_us });
            LOG_INF("bench: mode=%s seq_len=%d repeat=%d/%d phase=prefill done total_ms=%.3f moe_copy_ms=%.3f kv_h2d_ms=%.3f\n",
                    mode_label.c_str(), seq_len, rep + 1, bench.repeats,
                    (t_prefill_end - t_prefill_start)/1000.0,
                    st_prefill.time_us/1000.0,
                    kv_prefill_us/1000.0);

            cb.phase = "decode";
            cb.seq_len = seq_len;
            cb.repeat = rep;
            ggml_backend_moe_copy_stats_reset();
            ggml_backend_kv_copy_stats_reset();
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
            const uint64_t kv_decode_us = append_attention_kv_copy_records(mode_label, "decode", seq_len, rep, node_records);
            copy_records.push_back({ mode_label, "decode", seq_len, rep, st_decode.calls, st_decode.bytes, st_decode.time_us });
            LOG_INF("bench: mode=%s seq_len=%d repeat=%d/%d phase=decode done total_ms=%.3f moe_copy_ms=%.3f kv_h2d_ms=%.3f\n",
                    mode_label.c_str(), seq_len, rep + 1, bench.repeats,
                    (t_decode_end - t_decode_start)/1000.0,
                    st_decode.time_us/1000.0,
                    kv_decode_us/1000.0);
        }
    }

    ggml_backend_kv_copy_stats_set_enabled(false);
    ggml_backend_moe_copy_stats_set_enabled(false);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return true;
}

static bool collect_real_route_distributions(
        const bench_params & bench,
        common_params params,
        const std::string & attention_placement,
        std::vector<route_record> & routes,
        model_info & info) {
    if (bench.prompt_json.empty()) {
        return true;
    }

    const std::vector<std::string> prompts = load_aime_prompt_json(bench.prompt_json);
    if (prompts.empty()) {
        LOG_ERR("%s: no prompts found in %s\n", __func__, bench.prompt_json.c_str());
        return false;
    }

    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (model == nullptr) {
        LOG_ERR("%s: failed to load model for route collection\n", __func__);
        return false;
    }

    const std::string arch = meta_str(model, "general.architecture", "");
    info.n_layer  = llama_model_n_layer(model);
    info.n_embd   = llama_model_n_embd(model);
    info.n_expert = (int32_t) meta_i64(model, arch + ".expert_count", info.n_expert);
    info.n_ff_exp = (int32_t) meta_i64(model, arch + ".expert_feed_forward_length", info.n_ff_exp);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<std::vector<llama_token>> tokenized;
    tokenized.reserve(prompts.size());
    for (const std::string & prompt : prompts) {
        tokenized.push_back(common_tokenize(vocab, prompt, true, true));
    }

    callback_data cb;
    cb.profile_all_nodes = false;
    cb.collect_routes = true;
    cb.mode = "real_route";
    cb.phase = "prefill";
    cb.route_layer = bench.route_layer;

    params.cb_eval = eval_callback;
    params.cb_eval_user_data = &cb;

    llama_context_params ctx_params = common_context_params_to_llama(params);
    ctx_params.offload_attn = placement_attn_gpu(attention_placement) ? 1 : 0;
    ctx_params.n_seq_max = 1;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        LOG_ERR("%s: failed to create context for route collection\n", __func__);
        llama_model_free(model);
        return false;
    }

    auto * mem = llama_get_memory(ctx);
    llama_batch batch = llama_batch_init((int32_t) std::max<uint32_t>(1, llama_n_batch(ctx)), 0, 1);

    for (const int32_t seq_len : bench.micro_tokens) {
        if (seq_len > (int32_t) llama_n_ctx(ctx)) {
            LOG_WRN("%s: skipping route seq_len=%d because n_ctx=%u\n", __func__, seq_len, llama_n_ctx(ctx));
            continue;
        }

        int32_t collected = 0;
        for (size_t sample = 0; sample < tokenized.size(); ++sample) {
            if (bench.route_max_prompts > 0 && collected >= bench.route_max_prompts) {
                break;
            }
            const auto & toks = tokenized[sample];
            if ((int32_t) toks.size() < seq_len) {
                continue;
            }

            std::vector<llama_token> clipped(toks.begin(), toks.begin() + seq_len);
            std::vector<int64_t> counts((size_t) std::max(0, info.n_expert), 0);
            cb.seq_len = seq_len;
            cb.repeat = (int32_t) sample;
            cb.route_counts = &counts;

            llama_memory_clear(mem, false);
            LOG_INF("route: seq_len=%d sample=%zu layer=%d start\n", seq_len, sample, bench.route_layer);
            if (!decode_range(ctx, batch, clipped, 0, seq_len, false)) {
                LOG_ERR("%s: route collection failed for seq_len=%d sample=%zu\n", __func__, seq_len, sample);
                llama_batch_free(batch);
                llama_free(ctx);
                llama_model_free(model);
                return false;
            }

            routes.push_back({
                seq_len,
                (int32_t) sample,
                (int32_t) toks.size(),
                bench.route_layer,
                std::move(counts),
            });
            collected++;
            LOG_INF("route: seq_len=%d sample=%zu layer=%d done\n", seq_len, sample, bench.route_layer);
        }

        if (collected == 0) {
            LOG_WRN("%s: no prompt in %s has enough tokens for seq_len=%d\n", __func__, bench.prompt_json.c_str(), seq_len);
        }
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);

    LOG_INF("%s: collected %zu route distribution(s)\n", __func__, routes.size());
    return true;
}

static int64_t gguf_i64_or(const gguf_context * ctx, const std::string & key, int64_t fallback) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        return fallback;
    }

    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(ctx, id);
        case GGUF_TYPE_INT8:   return gguf_get_val_i8(ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(ctx, id);
        default:               return fallback;
    }
}

static std::string gguf_str_or(const gguf_context * ctx, const std::string & key, const std::string & fallback) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        return fallback;
    }
    const char * value = gguf_get_val_str(ctx, id);
    return value == nullptr ? fallback : value;
}

static bool load_model_info_metadata(const common_params & params, model_info & info) {
    gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };
    gguf_context * gguf = gguf_init_from_file(params.model.path.c_str(), gguf_params);
    if (gguf == nullptr) {
        LOG_WRN("%s: failed to load GGUF metadata; microbench will use explicit/default dimensions\n", __func__);
        return true;
    }

    const std::string arch = gguf_str_or(gguf, "general.architecture", "");
    if (!arch.empty()) {
        info.n_layer  = (int32_t) gguf_i64_or(gguf, arch + ".block_count", info.n_layer);
        info.n_embd   = (int32_t) gguf_i64_or(gguf, arch + ".embedding_length", info.n_embd);
        info.n_expert = (int32_t) gguf_i64_or(gguf, arch + ".expert_count", info.n_expert);
        info.n_ff_exp = (int32_t) gguf_i64_or(gguf, arch + ".expert_feed_forward_length", info.n_ff_exp);
    }

    gguf_free(gguf);
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

    const int32_t d_model = info.n_embd > 0 ? info.n_embd : bench.micro_d_model;
    const int32_t d_ff = info.n_ff_exp > 0 ? info.n_ff_exp : bench.micro_d_ff;
    const int32_t n_experts = 2;

    ggml_init_params init_params = {
        /* .mem_size   = */ 6*ggml_tensor_overhead() + 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx_src = ggml_init(init_params);
    ggml_context * ctx_dst = ggml_init(init_params);
    ggml_tensor * gate_src = ggml_new_tensor_3d(ctx_src, GGML_TYPE_F16, d_model, d_ff, n_experts);
    ggml_tensor * up_src   = ggml_new_tensor_3d(ctx_src, GGML_TYPE_F16, d_model, d_ff, n_experts);
    ggml_tensor * down_src = ggml_new_tensor_3d(ctx_src, GGML_TYPE_F16, d_ff, d_model, n_experts);
    ggml_tensor * gate_dst = ggml_new_tensor_3d(ctx_dst, GGML_TYPE_F16, d_model, d_ff, n_experts);
    ggml_tensor * up_dst   = ggml_new_tensor_3d(ctx_dst, GGML_TYPE_F16, d_model, d_ff, n_experts);
    ggml_tensor * down_dst = ggml_new_tensor_3d(ctx_dst, GGML_TYPE_F16, d_ff, d_model, n_experts);

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

    auto copy_one_expert = [&](ggml_tensor * dst, ggml_tensor * src, int32_t expert_id) -> uint64_t {
        const size_t expert_size = src->nb[2];
        const size_t expert_offset = (size_t) expert_id * expert_size;
        const size_t padding = std::min<size_t>(expert_size, 512);
        const size_t padding_end = expert_id < n_experts - 1 ? padding : 0;
        const size_t copy_size = expert_size + padding_end;

        const int64_t t0 = ggml_time_us();
        ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + expert_offset, expert_offset, copy_size);
        ggml_backend_synchronize(backend);
        const int64_t t1 = ggml_time_us();
        return (uint64_t) (t1 - t0);
    };

    auto full_expert_bytes = [&]() -> uint64_t {
        if (bench.expert_bytes > 0) {
            return (uint64_t) bench.expert_bytes;
        }
        uint64_t bytes = 0;
        for (ggml_tensor * t : { gate_src, up_src, down_src }) {
            const size_t expert_size = t->nb[2];
            bytes += expert_size + std::min<size_t>(expert_size, 512);
        }
        return bytes;
    };

    copy_one_expert(gate_dst, gate_src, 0);
    copy_one_expert(up_dst,   up_src,   0);
    copy_one_expert(down_dst, down_src, 0);

    for (int32_t rep = 0; rep < bench.repeats; ++rep) {
        uint64_t time_us = 0;
        time_us += copy_one_expert(gate_dst, gate_src, 0);
        time_us += copy_one_expert(up_dst,   up_src,   0);
        time_us += copy_one_expert(down_dst, down_src, 0);
        out.push_back({ "expert_h2d_pinned", "copy", 0, 0, 1, -1, 0, 0, 1, -1, rep, (int64_t) time_us, full_expert_bytes() });
    }

    ggml_backend_buffer_free(gpu_buf);
    ggml_backend_buffer_free(host_buf);
    ggml_free(ctx_dst);
    ggml_free(ctx_src);
    ggml_backend_free(backend);
    return true;
}

static ggml_cgraph * build_group_graph(ggml_context * ctx, int32_t d_model, int32_t d_ff, int32_t n_experts, int32_t n_tokens) {
    ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_model, d_ff, n_experts);
    ggml_tensor * up   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_model, d_ff, n_experts);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_ff, d_model, n_experts);
    ggml_tensor * x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_model, n_experts, n_tokens);
    ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_experts, n_tokens);

    ggml_tensor * ffn_gate = ggml_mul_mat_id(ctx, gate, x, ids);
    ggml_tensor * ffn_up   = ggml_mul_mat_id(ctx, up,   x, ids);
    ggml_tensor * act      = ggml_glu_split(ctx, ffn_gate, ffn_up, GGML_GLU_OP_SWIGLU);
    ggml_tensor * y        = ggml_mul_mat_id(ctx, down, act, ids);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    return gf;
}

static ggml_cgraph * build_serial_graph(ggml_context * ctx, int32_t d_model, int32_t d_ff, int32_t n_experts, int32_t n_tokens) {
    ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_model, d_ff, n_experts);
    ggml_tensor * up   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_model, d_ff, n_experts);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_ff, d_model, n_experts);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, n_tokens);
    ggml_tensor * acc = nullptr;
    for (int32_t i = 0; i < n_experts; ++i) {
        ggml_tensor * gate_i = ggml_view_2d(ctx, gate, d_model, d_ff, gate->nb[1], i*gate->nb[2]);
        ggml_tensor * up_i   = ggml_view_2d(ctx, up,   d_model, d_ff, up->nb[1],   i*up->nb[2]);
        ggml_tensor * down_i = ggml_view_2d(ctx, down, d_ff, d_model, down->nb[1], i*down->nb[2]);

        ggml_tensor * ffn_gate = ggml_mul_mat(ctx, gate_i, x);
        ggml_tensor * ffn_up   = ggml_mul_mat(ctx, up_i,   x);
        ggml_tensor * act      = ggml_glu_split(ctx, ffn_gate, ffn_up, GGML_GLU_OP_SWIGLU);
        ggml_tensor * yi       = ggml_mul_mat(ctx, down_i, act);
        acc = acc ? ggml_add(ctx, acc, yi) : yi;
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, acc);
    return gf;
}

static ggml_tensor * build_group_routed_path(
        ggml_context * ctx,
        int32_t d_model,
        int32_t d_ff,
        int32_t group_experts,
        const std::vector<int32_t> & token_counts,
        std::vector<int32_t> & ids_data) {
    int32_t total_tokens = 0;
    for (int32_t i = 0; i < group_experts; ++i) {
        total_tokens += token_counts[i];
    }
    if (group_experts <= 0 || total_tokens <= 0) {
        return nullptr;
    }

    ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_model, d_ff, group_experts);
    ggml_tensor * up   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_model, d_ff, group_experts);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_ff, d_model, group_experts);
    ggml_tensor * x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_model, 1, total_tokens);
    ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, total_tokens);
    ggml_set_name(ids, "micro_group_ids");

    ids_data.clear();
    ids_data.reserve((size_t) total_tokens);
    for (int32_t expert = 0; expert < group_experts; ++expert) {
        for (int32_t t = 0; t < token_counts[expert]; ++t) {
            ids_data.push_back(expert);
        }
    }

    ggml_tensor * ffn_gate = ggml_mul_mat_id(ctx, gate, x, ids);
    ggml_tensor * ffn_up   = ggml_mul_mat_id(ctx, up,   x, ids);
    ggml_tensor * act      = ggml_glu_split(ctx, ffn_gate, ffn_up, GGML_GLU_OP_SWIGLU);
    return ggml_mul_mat_id(ctx, down, act, ids);
}

static void build_serial_routed_paths(
        ggml_context * ctx,
        ggml_cgraph * gf,
        int32_t d_model,
        int32_t d_ff,
        int32_t first_expert,
        const std::vector<int32_t> & token_counts) {
    const int32_t serial_experts = (int32_t) token_counts.size() - first_expert;
    if (serial_experts <= 0) {
        return;
    }

    ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_model, d_ff, serial_experts);
    ggml_tensor * up   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_model, d_ff, serial_experts);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_ff, d_model, serial_experts);

    for (int32_t expert = first_expert; expert < (int32_t) token_counts.size(); ++expert) {
        const int32_t n_tokens = token_counts[expert];
        if (n_tokens <= 0) {
            continue;
        }
        const int32_t local = expert - first_expert;
        ggml_tensor * x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, n_tokens);
        ggml_tensor * gate_i = ggml_view_2d(ctx, gate, d_model, d_ff, gate->nb[1], local*gate->nb[2]);
        ggml_tensor * up_i   = ggml_view_2d(ctx, up,   d_model, d_ff, up->nb[1],   local*up->nb[2]);
        ggml_tensor * down_i = ggml_view_2d(ctx, down, d_ff, d_model, down->nb[1], local*down->nb[2]);

        ggml_tensor * ffn_gate = ggml_mul_mat(ctx, gate_i, x);
        ggml_tensor * ffn_up   = ggml_mul_mat(ctx, up_i,   x);
        ggml_tensor * act      = ggml_glu_split(ctx, ffn_gate, ffn_up, GGML_GLU_OP_SWIGLU);
        ggml_tensor * y        = ggml_mul_mat(ctx, down_i, act);
        ggml_build_forward_expand(gf, y);
    }
}

static ggml_cgraph * build_alpha_gemm_graph(
        ggml_context * ctx,
        int32_t d_model,
        int32_t d_ff,
        int32_t group_experts,
        const std::vector<int32_t> & token_counts,
        std::vector<int32_t> & ids_data) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    if (ggml_tensor * y_group = build_group_routed_path(ctx, d_model, d_ff, group_experts, token_counts, ids_data)) {
        ggml_build_forward_expand(gf, y_group);
    }
    build_serial_routed_paths(ctx, gf, d_model, d_ff, group_experts, token_counts);
    return gf;
}

static std::vector<int32_t> make_prefill_token_counts(int32_t total_tokens, int32_t active_experts) {
    std::vector<int32_t> counts((size_t) active_experts, 0);
    if (total_tokens <= 0 || active_experts <= 0) {
        return counts;
    }

    const int32_t base_experts = std::min(total_tokens, active_experts);
    for (int32_t i = 0; i < base_experts; ++i) {
        counts[i] = 1;
    }
    if (total_tokens <= active_experts) {
        return counts;
    }

    const int64_t remaining = (int64_t) total_tokens - active_experts;
    const int64_t weight_sum = (int64_t) active_experts * (active_experts + 1) / 2;
    int64_t assigned = 0;
    std::vector<std::pair<int64_t, int32_t>> residuals;
    residuals.reserve((size_t) active_experts);
    for (int32_t i = 0; i < active_experts; ++i) {
        const int64_t weight = active_experts - i;
        const int64_t weighted = remaining * weight;
        const int32_t add = (int32_t) (weighted / weight_sum);
        counts[i] += add;
        assigned += add;
        residuals.push_back({ weighted % weight_sum, i });
    }

    std::sort(residuals.begin(), residuals.end(), [](const auto & a, const auto & b) {
        if (a.first != b.first) {
            return a.first > b.first;
        }
        return a.second < b.second;
    });
    for (int64_t left = remaining - assigned; left > 0; --left) {
        counts[residuals[(size_t) ((remaining - assigned - left) % residuals.size())].second]++;
    }
    return counts;
}

static std::vector<int32_t> make_decode_token_counts(int32_t active_experts) {
    return std::vector<int32_t>((size_t) active_experts, 1);
}

static bool fill_graph_inputs(ggml_context * ctx, const std::vector<int32_t> * ids_override = nullptr) {
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
            if (ids_override != nullptr && ids_override->size() == ids.size()) {
                ids = *ids_override;
            } else {
                for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                        ids[(size_t) (i1*t->ne[0] + i0)] = (int32_t) i0;
                    }
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
        int32_t label_tokens,
        int32_t active_tokens,
        int32_t experts,
        const bench_params & bench,
        const model_info & info,
        std::vector<micro_record> & out) {
    const int32_t d_model = info.n_embd > 0 ? info.n_embd : bench.micro_d_model;
    const int32_t d_ff = info.n_ff_exp > 0 ? info.n_ff_exp : bench.micro_d_ff;
    const int32_t n_experts = experts;

    const size_t mem_size = 64ULL*1024ULL*1024ULL;
    ggml_init_params init_params = {
        /* .mem_size   = */ mem_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(init_params);
    ggml_cgraph * gf = group ?
        build_group_graph(ctx, d_model, d_ff, n_experts, active_tokens) :
        build_serial_graph(ctx, d_model, d_ff, n_experts, active_tokens);

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
        out.push_back({
            group ? "moe_group_gemm" : "moe_serial_gemm",
            phase,
            label_tokens,
            active_tokens,
            experts,
            -1,
            group ? experts : 0,
            group ? 0 : experts,
            experts,
            -1,
            rep,
            t1 - t0,
            0,
        });
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static bool run_one_alpha_gemm_micro(
        ggml_backend_t backend,
        const std::string & phase,
        int32_t label_tokens,
        int32_t alpha_pct,
        int32_t sample,
        const std::vector<int32_t> & token_counts,
        const bench_params & bench,
        const model_info & info,
        std::vector<micro_record> & out) {
    const int32_t d_model = info.n_embd > 0 ? info.n_embd : bench.micro_d_model;
    const int32_t d_ff = info.n_ff_exp > 0 ? info.n_ff_exp : bench.micro_d_ff;
    const int32_t active_experts = (int32_t) token_counts.size();
    const int32_t group_experts = active_experts * alpha_pct / 100;
    const int32_t serial_experts = active_experts - group_experts;
    const int32_t active_tokens = std::accumulate(token_counts.begin(), token_counts.end(), 0);

    if (active_experts <= 0 || active_tokens <= 0) {
        return true;
    }

    const size_t mem_size = 256ULL*1024ULL*1024ULL;
    ggml_init_params init_params = {
        /* .mem_size   = */ mem_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    std::vector<int32_t> ids_data;
    ggml_context * ctx = ggml_init(init_params);
    ggml_cgraph * gf = build_alpha_gemm_graph(ctx, d_model, d_ff, group_experts, token_counts, ids_data);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        LOG_WRN("%s: failed to allocate alpha microbench tensors for %s/alpha%d\n", __func__, phase.c_str(), alpha_pct);
        ggml_free(ctx);
        return true;
    }

    fill_graph_inputs(ctx, ids_data.empty() ? nullptr : &ids_data);

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        LOG_WRN("%s: warmup failed for %s/alpha%d\n", __func__, phase.c_str(), alpha_pct);
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
            LOG_WRN("%s: microbench compute failed for %s/alpha%d\n", __func__, phase.c_str(), alpha_pct);
            break;
        }
        out.push_back({
            "moe_gemm",
            phase,
            label_tokens,
            active_tokens,
            active_experts,
            alpha_pct,
            group_experts,
            serial_experts,
            active_experts,
            sample,
            rep,
            t1 - t0,
            0,
        });
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static std::vector<int32_t> route_counts_i32(const route_record & route) {
    std::vector<int32_t> counts;
    counts.reserve(route.expert_counts.size());
    for (const int64_t count : route.expert_counts) {
        counts.push_back((int32_t) std::min<int64_t>(count, INT32_MAX));
    }
    return counts;
}

static bool run_gemm_microbench(
        const bench_params & bench,
        const model_info & info,
        const std::vector<route_record> & routes,
        std::vector<micro_record> & out) {
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

    const int32_t prefill_experts = info.n_expert > 0 ? info.n_expert : bench.micro_experts;
    const int32_t decode_experts = 8;
    for (const int32_t alpha_pct : bench.micro_alpha_pcts) {
        if (!routes.empty()) {
            for (const route_record & route : routes) {
                run_one_alpha_gemm_micro(
                        backend,
                        "prefill",
                        route.seq_len,
                        alpha_pct,
                        route.sample,
                        route_counts_i32(route),
                        bench,
                        info,
                        out);
            }
        } else {
            for (const int32_t tokens : bench.micro_tokens) {
                run_one_alpha_gemm_micro(
                        backend,
                        "prefill",
                        tokens,
                        alpha_pct,
                        -1,
                        make_prefill_token_counts(tokens, prefill_experts),
                        bench,
                        info,
                        out);
            }
        }
        run_one_alpha_gemm_micro(
                backend,
                "decode",
                bench.micro_decode_tokens,
                alpha_pct,
                -1,
                make_decode_token_counts(decode_experts),
                bench,
                info,
                out);
    }

    ggml_backend_free(backend);
    return true;
}

static void write_raw_jsonl(
        const std::filesystem::path & path,
        const std::vector<node_record> & nodes,
        const std::vector<copy_record> & copies,
        const std::vector<route_record> & routes,
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
    for (const auto & rec : routes) {
        jsonl << "{\"kind\":\"moe_route_distribution\""
              << ",\"phase\":\"prefill\""
              << ",\"seq_len\":" << rec.seq_len
              << ",\"sample\":" << rec.sample
              << ",\"original_tokens\":" << rec.original_tokens
              << ",\"layer\":" << rec.layer
              << ",\"expert_counts\":[";
        for (size_t i = 0; i < rec.expert_counts.size(); ++i) {
            if (i) {
                jsonl << ",";
            }
            jsonl << rec.expert_counts[i];
        }
        jsonl << "]}\n";
    }
    for (const auto & rec : micros) {
        jsonl << "{\"kind\":\"" << json_escape(rec.kind) << "\""
              << ",\"phase\":\"" << json_escape(rec.phase) << "\""
              << ",\"tokens\":" << rec.tokens
              << ",\"active_tokens\":" << rec.active_tokens
              << ",\"experts\":" << rec.experts
              << ",\"alpha_pct\":" << rec.alpha_pct
              << ",\"group_experts\":" << rec.group_experts
              << ",\"serial_experts\":" << rec.serial_experts
              << ",\"active_experts\":" << rec.active_experts
              << ",\"sample\":" << rec.sample
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
        const std::vector<micro_record> & micros,
        int32_t decode_tokens) {
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
        double value = (double) it.second;
        if (it.first.phase == "decode" && decode_tokens > 0) {
            value /= (double) decode_tokens;
        }
        buckets[key.str()].values.push_back(value);
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
        if (rec.alpha_pct >= 0) {
            key << rec.kind << ",micro_alpha" << rec.alpha_pct << "," << rec.phase << "," << rec.tokens << ",-1,time_us";
        } else {
            key << rec.kind << ",micro_h" << rec.experts << "," << rec.phase << "," << rec.tokens << ",-1,time_us";
        }
        buckets[key.str()].values.push_back((double) rec.time_us);
        if (rec.bytes > 0) {
            std::ostringstream kb;
            if (rec.alpha_pct >= 0) {
                kb << rec.kind << ",micro_alpha" << rec.alpha_pct << "," << rec.phase << "," << rec.tokens << ",-1,bytes";
            } else {
                kb << rec.kind << ",micro_h" << rec.experts << "," << rec.phase << "," << rec.tokens << ",-1,bytes";
            }
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
    report << "- attention_placement: `" << bench.attention_placement << "`\n";
    report << "- attention_backend_compat: `" << bench.attention_backend << "`\n";
    report << "- prompt_json: `" << (bench.prompt_json.empty() ? "-" : bench.prompt_json) << "`\n";
    report << "- route_layer: `" << bench.route_layer << "`\n";
    report << "- route_max_prompts: `" << bench.route_max_prompts << "`\n";
    report << "- repeats: `" << bench.repeats << "`\n";
    report << "- decode_tokens: `" << bench.decode_tokens << "`\n";
    report << "- skip_microbench: `" << (bench.skip_microbench ? 1 : 0) << "`\n";
    report << "- skip_gemm_microbench: `" << (bench.skip_gemm_microbench ? 1 : 0) << "`\n";
    report << "- profile_all_nodes: `" << (bench.profile_all_nodes ? 1 : 0) << "`\n";
    report << "- micro tokens: `";
    for (size_t i = 0; i < bench.micro_tokens.size(); ++i) {
        if (i) {
            report << ",";
        }
        report << bench.micro_tokens[i];
    }
    report << "`\n";
    report << "- micro experts sweep: `";
    for (size_t i = 0; i < bench.micro_experts_sweep.size(); ++i) {
        if (i) {
            report << ",";
        }
        report << bench.micro_experts_sweep[i];
    }
    report << "`\n";
    report << "- micro alpha pcts: `";
    for (size_t i = 0; i < bench.micro_alpha_pcts.size(); ++i) {
        if (i) {
            report << ",";
        }
        report << bench.micro_alpha_pcts[i];
    }
    report << "`\n";
    report << "- micro prefill fallback experts: `" << bench.micro_experts << "`\n";
    report << "- micro decode active tokens: `" << bench.micro_decode_tokens << "`\n";
    report << "- model layers: `" << info.n_layer << "`\n";
    report << "- model embedding: `" << info.n_embd << "`\n";
    report << "- experts: `" << info.n_expert << "`\n";
    report << "- expert FFN: `" << info.n_ff_exp << "`\n\n";
    report << "## Outputs\n\n";
    report << "- `results.jsonl`: raw node, attention_kv_h2d, attention-layer, runtime MoE-copy, expert_h2d_pinned, and microbench records\n";
    report << "- `summary.csv`: aggregated averages and stddevs; decode attention_layer rows are per-token averages\n";
    report << "- `plots/*.svg`: generated by `plot_qwen3vl_moe_bench.py`\n\n";
    report << "Note: runtime MoE-copy and CPU-KV-to-GPU-attention profiling synchronize the destination backend around measured copies, so they are for measurement rather than maximum-throughput serving.\n";
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
    if (bench.mode == "moe_cpu_offload" && any_placement_attn_gpu(bench) &&
            params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }

    int32_t max_len = *std::max_element(bench.lengths.begin(), bench.lengths.end());
    if (!bench.prompt_json.empty() && !bench.micro_tokens.empty()) {
        max_len = std::max(max_len, *std::max_element(bench.micro_tokens.begin(), bench.micro_tokens.end()));
    }
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
    std::vector<route_record> route_records;
    std::vector<micro_record> micro_records;
    model_info info;
    bool ok = true;

    if (!bench.skip_model_bench) {
        const std::vector<std::string> placements = attention_placements_to_run(bench);
        if (bench.mode == "moe_cpu_sweep") {
            for (const std::string & placement : placements) {
                for (int32_t n_cpu_moe : bench.n_cpu_moe_sweep) {
                    std::vector<std::string> owned_patterns;
                    std::string mode_label = make_mode_label(placement, "moe_cpu_sweep", n_cpu_moe);
                    common_params run_params = make_run_params(params, bench, "moe_cpu_sweep", placement, n_cpu_moe, owned_patterns);
                    ok = run_model_bench(bench, run_params, mode_label, placement, node_records, copy_records, info) && ok;
                }
            }
        } else {
            for (const std::string & placement : placements) {
                std::vector<std::string> owned_patterns;
                common_params run_params = make_run_params(params, bench, bench.mode, placement, bench.n_cpu_moe_layers, owned_patterns);
                ok = run_model_bench(bench, run_params, make_mode_label(placement, bench.mode, bench.n_cpu_moe_layers), placement, node_records, copy_records, info) && ok;
            }
        }
    }

    if (!bench.skip_microbench) {
        if ((info.n_embd <= 0 || info.n_ff_exp <= 0) && !params.model.path.empty()) {
            ok = load_model_info_metadata(params, info) && ok;
        }
        if (!bench.prompt_json.empty()) {
            std::vector<std::string> owned_patterns;
            const std::string placement = attention_placements_to_run(bench).front();
            common_params route_params = make_run_params(params, bench, bench.mode, placement, bench.n_cpu_moe_layers, owned_patterns);
            ok = collect_real_route_distributions(bench, route_params, placement, route_records, info) && ok;
        }
        ok = run_h2d_microbench(bench, info, micro_records) && ok;
        if (!bench.skip_gemm_microbench) {
            ok = run_gemm_microbench(bench, info, route_records, micro_records) && ok;
        }
    }

    const std::filesystem::path out_dir = bench.out_dir;
    write_raw_jsonl(out_dir / "results.jsonl", node_records, copy_records, route_records, micro_records);
    write_summary_csv(out_dir / "summary.csv", node_records, copy_records, micro_records, bench.decode_tokens);
    write_report(out_dir / "report.md", bench, info, argc, argv);

    llama_backend_free();

    LOG_INF("wrote %s, %s, %s\n",
            (out_dir / "results.jsonl").string().c_str(),
            (out_dir / "summary.csv").string().c_str(),
            (out_dir / "report.md").string().c_str());

    return ok ? 0 : 1;
}
