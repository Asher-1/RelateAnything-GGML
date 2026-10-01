#include "relateanything/full_model.hpp"

#include "relateanything/backend.hpp"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace relateanything {
namespace {

constexpr char kImageMagic[] = "RAIMv1\0";
constexpr char kImageOutputMagic[] = "RAFOv1\0";
constexpr float kNegInf = -1.0e9f;

template <typename T>
void read_exact(std::ifstream & stream, T * dst, std::size_t count = 1) {
    const std::size_t bytes = sizeof(T) * count;
    stream.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(bytes));
    if (!stream || static_cast<std::size_t>(stream.gcount()) != bytes) {
        throw std::runtime_error("truncated RAIM input");
    }
}

std::size_t product(std::size_t a, std::size_t b, const char * what) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::runtime_error(std::string(what) + " is too large");
    }
    return a * b;
}

struct TensorLoader {
    Options options;
    std::unique_ptr<Backend> backend;
    gguf_context * gguf = nullptr;
    ggml_context * weights_ctx = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
    std::map<std::string, std::string> original_to_emitted;
    std::filesystem::path path;

    explicit TensorLoader(const Options & opts) : options(opts), backend(std::make_unique<Backend>(opts)), path(opts.model) {
        try {
            if (path.empty()) throw std::runtime_error("full graph model path is required");
            gguf_init_params params{};
            params.no_alloc = true;
            params.ctx = &weights_ctx;
            gguf = gguf_init_from_file(path.string().c_str(), params);
            if (!gguf || !weights_ctx) throw std::runtime_error("cannot load full graph GGUF: " + path.string());
            const auto graph_id = gguf_find_key(gguf, "ra.graph_kind");
            if (graph_id < 0 || gguf_get_kv_type(gguf, graph_id) != GGUF_TYPE_STRING ||
                std::string(gguf_get_val_str(gguf, graph_id)) != "relateanything_full_v1")
                throw std::runtime_error("image inference requires a relateanything_full_v1 model");

            const int64_t map_id = gguf_find_key(gguf, "ra.tensor_name_map");
            if (map_id >= 0 && gguf_get_arr_type(gguf, map_id) == GGUF_TYPE_STRING) {
                const std::size_t n = gguf_get_arr_n(gguf, map_id);
                for (std::size_t i = 0; i + 1 < n; i += 2) {
                    original_to_emitted.emplace(gguf_get_arr_str(gguf, map_id, i),
                                                gguf_get_arr_str(gguf, map_id, i + 1));
                }
            }
            for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
                const char * emitted = gguf_get_tensor_name(gguf, i);
                tensors.emplace(emitted, ggml_get_tensor(weights_ctx, emitted));
            }
            weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, backend->accelerator);
            if (!weights_buffer) throw std::runtime_error("cannot allocate full graph weights");
            ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::ifstream stream(path, std::ios::binary);
            if (!stream) throw std::runtime_error("cannot read full graph GGUF: " + path.string());
            for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
                const char * emitted = gguf_get_tensor_name(gguf, i);
                ggml_tensor * tensor = ggml_get_tensor(weights_ctx, emitted);
                const std::size_t offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, i);
                const std::size_t bytes = ggml_nbytes(tensor);
                std::vector<std::uint8_t> data(bytes);
                stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
                read_exact(stream, data.data(), bytes);
                ggml_backend_tensor_set(tensor, data.data(), 0, bytes);
            }
        } catch (...) {
            if (weights_buffer) ggml_backend_buffer_free(weights_buffer);
            if (weights_ctx) ggml_free(weights_ctx);
            if (gguf) gguf_free(gguf);
            throw;
        }
    }

    ~TensorLoader() {
        if (weights_buffer) ggml_backend_buffer_free(weights_buffer);
        if (weights_ctx) ggml_free(weights_ctx);
        if (gguf) gguf_free(gguf);
    }

    ggml_tensor * tensor(const std::string & original) const {
        std::string emitted = original;
        if (const auto it = original_to_emitted.find(original); it != original_to_emitted.end()) emitted = it->second;
        const auto it = tensors.find(emitted);
        if (it == tensors.end() || !it->second) throw std::runtime_error("missing full graph tensor: " + original);
        return it->second;
    }

    std::size_t meta_u32(const char * key, std::size_t fallback = 0) const {
        const int64_t id = gguf_find_key(gguf, key);
        if (id < 0) return fallback;
        switch (gguf_get_kv_type(gguf, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(gguf, id);
        case GGUF_TYPE_INT32: return static_cast<std::size_t>(gguf_get_val_i32(gguf, id));
        default: return fallback;
        }
    }

    float meta_float(const char * key, float fallback) const {
        const int64_t id = gguf_find_key(gguf, key);
        if (id < 0) return fallback;
        if (gguf_get_kv_type(gguf, id) == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(gguf, id);
        return fallback;
    }

    std::vector<float> host_f32(ggml_tensor * tensor) const {
        if (!tensor) throw std::runtime_error("cannot read null tensor");
        if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
            throw std::runtime_error("host_f32 is only used for F32/F16 auxiliary tensors: " + std::string(tensor->name));
        }
        const std::size_t n = ggml_nelements(tensor);
        std::vector<std::uint8_t> raw(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
        std::vector<float> out(n);
        if (tensor->type == GGML_TYPE_F32) std::memcpy(out.data(), raw.data(), n * sizeof(float));
        else {
            const auto * p = reinterpret_cast<const std::uint16_t *>(raw.data());
            for (std::size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(p[i]);
        }
        return out;
    }
};

struct GraphRun {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    std::vector<std::pair<ggml_tensor *, std::vector<std::uint8_t>>> inputs;
    std::size_t input_bytes = 0;
    GraphRun() = default;
    GraphRun(const GraphRun &) = delete;
    GraphRun & operator=(const GraphRun &) = delete;
    GraphRun(GraphRun && other) noexcept : ctx(other.ctx), graph(other.graph), inputs(std::move(other.inputs)) { other.ctx = nullptr; }
    ~GraphRun() { if (ctx) ggml_free(ctx); }
};

GraphRun make_run() {
    GraphRun run;
    const std::size_t bytes = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
    run.ctx = ggml_init({bytes, nullptr, true});
    if (!run.ctx) throw std::runtime_error("cannot create full graph context");
    return run;
}

ggml_tensor * input_tensor(GraphRun & run, ggml_context * ctx, enum ggml_type type,
                           int64_t ne0, int64_t ne1, int64_t ne2 = 1, int64_t ne3 = 1,
                           const void * data = nullptr) {
    ggml_tensor * t = nullptr;
    if (ne3 != 1) t = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
    else if (ne2 != 1) t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
    else if (ne1 != 1) t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    else t = ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_input(t);
    if (data) {
        const auto * ptr = static_cast<const std::uint8_t *>(data);
        run.inputs.emplace_back(t, std::vector<std::uint8_t>(ptr, ptr + ggml_nbytes(t)));
    }
    return t;
}

ggml_tensor * as_f32(ggml_context * ctx, ggml_tensor * tensor) {
    return tensor && tensor->type != GGML_TYPE_F32 ? ggml_cast(ctx, tensor, GGML_TYPE_F32) : tensor;
}

ggml_tensor * matmul(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * x) {
    auto * y = ggml_mul_mat(ctx, weight, x);
    // Accumulate in F32 for all storage types. Otherwise Vulkan may accumulate
    // attention and linear layers in F16, compounding rounding at every layer.
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    return y;
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias) {
    ggml_tensor * y = matmul(ctx, weight, x);
    return bias ? ggml_add(ctx, y, as_f32(ctx, bias)) : y;
}

ggml_tensor * norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight,
                   ggml_tensor * bias, float eps) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, as_f32(ctx, weight));
    return bias ? ggml_add(ctx, y, as_f32(ctx, bias)) : y;
}

ggml_tensor * activation(ggml_context * ctx, ggml_tensor * x, bool silu) {
    return silu ? ggml_silu(ctx, x) : ggml_gelu_erf(ctx, x);
}

ggml_tensor * concat_axis1(ggml_context * ctx, const std::vector<ggml_tensor *> & xs) {
    ggml_tensor * out = as_f32(ctx, xs.front());
    for (std::size_t i = 1; i < xs.size(); ++i) out = ggml_concat(ctx, out, as_f32(ctx, xs[i]), 1);
    return out;
}

ggml_tensor * rope(ggml_context * ctx, ggml_tensor * x, ggml_tensor * cos, ggml_tensor * sin, int64_t dh) {
    // x is [dh, tokens, heads, batch], contiguous after the permutation.
    const int64_t half = dh / 2;
    ggml_tensor * a = ggml_view_4d(ctx, x, half, x->ne[1], x->ne[2], x->ne[3],
                                   x->nb[1], x->nb[2], x->nb[3], 0);
    ggml_tensor * b = ggml_view_4d(ctx, x, half, x->ne[1], x->ne[2], x->ne[3],
                                   x->nb[1], x->nb[2], x->nb[3], static_cast<std::size_t>(half) * sizeof(float));
    ggml_tensor * neg_b = ggml_neg(ctx, ggml_cont(ctx, b));
    ggml_tensor * rotated = ggml_concat(ctx, neg_b, a, 0);
    return ggml_add(ctx, ggml_mul(ctx, x, cos), ggml_mul(ctx, rotated, sin));
}

void build_mask(std::vector<float> & mask, std::size_t keys, std::size_t queries,
                const std::vector<std::uint8_t> & valid_keys) {
    mask.assign(keys * queries, 0.0f);
    for (std::size_t k = 0; k < keys; ++k) if (!valid_keys[k])
        for (std::size_t q = 0; q < queries; ++q) mask[k + q * keys] = kNegInf;
}

std::vector<float> fourier_point(float x, float y, int freqs = 16) {
    static const std::array<float, 16> frequencies = [] {
        std::array<float, 16> f{};
        for (int i = 0; i < 16; ++i) f[i] = std::pow(2.0f, i * 7.0f / 15.0f) * static_cast<float>(M_PI);
        return f;
    }();
    if (freqs != 16) throw std::runtime_error("unsupported Fourier frequency count");
    std::vector<float> out;
    out.reserve(4 * freqs);
    for (int axis = 0; axis < 2; ++axis) {
        const float value = axis == 0 ? x : y;
        for (int i = 0; i < freqs; ++i) out.push_back(std::sin(value * frequencies[i]));
        for (int i = 0; i < freqs; ++i) out.push_back(std::cos(value * frequencies[i]));
    }
    return out;
}

std::array<float, 4> xyxy(const float * b) {
    const float x1 = std::max(0.0f, b[0] - b[2] * 0.5f);
    const float y1 = std::max(0.0f, b[1] - b[3] * 0.5f);
    const float x2 = std::min(1.0f, b[0] + b[2] * 0.5f);
    const float y2 = std::min(1.0f, b[1] + b[3] * 0.5f);
    return {x1, y1, x2, y2};
}

std::array<float, 19> geometry(const float * s, const float * o) {
    const float eps = 1e-6f;
    const float dx = (o[0] - s[0]) / (s[2] + eps);
    const float dy = (o[1] - s[1]) / (s[3] + eps);
    const float log_wr = std::log((o[2] + eps) / (s[2] + eps));
    const float log_hr = std::log((o[3] + eps) / (s[3] + eps));
    const float sa_raw = s[2] * s[3], oa_raw = o[2] * o[3];
    const float sa = std::max(sa_raw, eps), oa = std::max(oa_raw, eps);
    const std::array<float, 4> sb = {s[0]-s[2]*0.5f, s[1]-s[3]*0.5f,
                                     s[0]+s[2]*0.5f, s[1]+s[3]*0.5f};
    const std::array<float, 4> ob = {o[0]-o[2]*0.5f, o[1]-o[3]*0.5f,
                                     o[0]+o[2]*0.5f, o[1]+o[3]*0.5f};
    const float ix = std::max(0.0f, std::min(sb[2], ob[2]) - std::max(sb[0], ob[0]));
    const float iy = std::max(0.0f, std::min(sb[3], ob[3]) - std::max(sb[1], ob[1]));
    const float inter = ix * iy, uni = sa + oa - inter + eps;
    const float dist = std::sqrt(std::max((o[0]-s[0])*(o[0]-s[0]) +
                                          (o[1]-s[1])*(o[1]-s[1]), eps));
    const float v[19] = {dx, dy, log_wr, log_hr, std::log((oa_raw + eps) / (sa_raw + eps)),
        std::log(sa_raw + eps), std::log(oa_raw + eps), inter / uni, inter / sa,
        inter / oa, std::log(std::max(s[2] / (s[3] + eps), eps)),
        std::log(std::max(o[2] / (o[3] + eps), eps)),
        (o[0] - s[0]) / dist, (o[1] - s[1]) / dist, o[1] - s[1],
        1.0f, 1.0f, inter / uni, inter / std::min(sa, oa)};
    std::array<float, 19> out{};
    for (int i = 0; i < 19; ++i) out[i] = 10.0f * std::tanh(v[i] / 10.0f);
    return out;
}

std::vector<float> read_f32(ggml_tensor * tensor) {
    std::vector<float> out(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(float));
    return out;
}

ggml_tensor * packed_view(ggml_context * ctx, ggml_tensor * weight,
                          int segment, int64_t width) {
    return ggml_view_2d(ctx, weight, weight->ne[0], width, weight->nb[1],
                        static_cast<std::size_t>(segment) * width * weight->nb[1]);
}

ggml_tensor * packed_bias(ggml_context * ctx, ggml_tensor * bias,
                          int segment, int64_t width) {
    return ggml_view_1d(ctx, bias, width,
                        static_cast<std::size_t>(segment) * width * ggml_element_size(bias));
}

ggml_tensor * packed_attention(ggml_context * ctx, ggml_tensor * query,
                               ggml_tensor * key, ggml_tensor * value,
                               ggml_tensor * in_weight, ggml_tensor * in_bias,
                               ggml_tensor * out_weight, ggml_tensor * out_bias,
                               int64_t dim, int64_t nheads, ggml_tensor * mask = nullptr,
                               bool flash = false) {
    const int64_t dh = dim / nheads;
    ggml_tensor * q = nullptr, * k = nullptr, * v = nullptr;
    if (query == key && key == value) {
        auto * qkv = linear(ctx, query, in_weight, in_bias);
        q = ggml_view_2d(ctx, qkv, dim, query->ne[1], qkv->nb[1], 0);
        k = ggml_view_2d(ctx, qkv, dim, key->ne[1], qkv->nb[1], dim * sizeof(float));
        v = ggml_view_2d(ctx, qkv, dim, value->ne[1], qkv->nb[1], 2 * dim * sizeof(float));
    } else {
        q = linear(ctx, query, packed_view(ctx, in_weight, 0, dim), packed_bias(ctx, in_bias, 0, dim));
        auto * kv_weight = ggml_view_2d(ctx, in_weight, dim, 2*dim, in_weight->nb[1], dim*in_weight->nb[1]);
        auto * kv_bias = ggml_view_1d(ctx, in_bias, 2*dim, dim*ggml_element_size(in_bias));
        auto * kv = linear(ctx, key, kv_weight, kv_bias);
        k = ggml_view_2d(ctx, kv, dim, key->ne[1], kv->nb[1], 0);
        v = ggml_view_2d(ctx, kv, dim, value->ne[1], kv->nb[1], dim*sizeof(float));
    }
    const int64_t nq = query->ne[1], nk = key->ne[1];
    auto split = [&](ggml_tensor * z) {
        return ggml_permute(ctx,
            ggml_reshape_4d(ctx, ggml_cont(ctx, z), dh, nheads, z->ne[1], 1),
            0, 2, 1, 3);
    };
    q = split(q); k = split(k); v = split(v);
    if (flash && dh % 32 == 0) {
        if (mask) {
            const int pad = (64 - mask->ne[1] % 64) % 64;
            mask = ggml_cast(ctx, pad ? ggml_pad(ctx, mask, 0, pad, 0, 0) : mask, GGML_TYPE_F16);
        }
        auto * attn = ggml_flash_attn_ext(ctx, ggml_cont(ctx, q),
            ggml_cast(ctx, k, GGML_TYPE_F16), ggml_cast(ctx, v, GGML_TYPE_F16),
            mask, 1.0f / std::sqrt(static_cast<float>(dh)), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        return linear(ctx, ggml_reshape_3d(ctx, attn, dim, nq, 1), out_weight, out_bias);
    }
    ggml_tensor * scores = matmul(ctx, k, q);
    ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, mask,
                                            1.0f / std::sqrt(static_cast<float>(dh)), 0.0f);
    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    ggml_tensor * attn = matmul(ctx, vt, probs);
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_3d(ctx, attn, dim, nq, 1);
    (void) nk;
    return linear(ctx, attn, out_weight, out_bias);
}

std::vector<float> make_fourier_matrix(const std::vector<float> & points,
                                       std::size_t count, int freqs = 16) {
    std::vector<float> out(4 * freqs * count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto f = fourier_point(points[2 * i], points[2 * i + 1], freqs);
        for (std::size_t j = 0; j < f.size(); ++j) out[j + f.size() * i] = f[j];
    }
    return out;
}

} // namespace

struct FullModel::Impl : TensorLoader {
    struct PackedAttention {
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        std::vector<ggml_tensor *> weights, biases;
        ~PackedAttention() {
            if (buffer) ggml_backend_buffer_free(buffer);
            if (ctx) ggml_free(ctx);
        }
    } packed_qkv;
    std::size_t c = 0, d = 0, depth = 0, heads = 0, intermediate = 0, patch = 16, regs = 4, predicates = 0;
    float eps = 1e-5f;
    bool swiglu = false;
    std::vector<float> vocabulary;
    std::vector<float> tap_weights, compose_gates;
    std::vector<std::string> predicate_names;
    float calibration_a = 1, calibration_b = 0;
    float vocabulary_scale = 1;
    std::vector<float> scene_fourier;
    std::vector<float> rope_cos, rope_sin;
    int rope_height = 0, rope_width = 0;
    std::size_t backend_buffer_bytes = 0;

    explicit Impl(const Options & options) : TensorLoader(options) {
        c = meta_u32("ra.hidden_size"); d = meta_u32("ra.relation_dim"); depth = meta_u32("ra.depth");
        heads = meta_u32("ra.num_heads"); intermediate = meta_u32("ra.intermediate_size"); patch = meta_u32("ra.patch_size", 16);
        regs = meta_u32("ra.num_register_tokens", 4); predicates = meta_u32("ra.predicate_count");
        eps = meta_float("ra.layer_norm_eps", 1e-5f);
        const int64_t ffn = gguf_find_key(gguf, "ra.ffn_type");
        swiglu = ffn >= 0 && std::string(gguf_get_val_str(gguf, ffn)) == "swiglu";
        if (!c || !d || !depth || !heads || !intermediate || !predicates) throw std::runtime_error("incomplete full graph metadata");
        if (c % heads || d % 8) throw std::runtime_error("invalid full graph dimensions");
        tap_weights = host_f32(t("ra.backbone.layer_weights"));
        compose_gates = host_f32(t("ra.compose_gate"));
        vocabulary_scale = std::min(std::exp(host_f32(t("ra.vocab_head.logit_scale"))[0]), 100.0f);
        calibration_a = meta_float("ra.calibration_a", 1.0f);
        calibration_b = meta_float("ra.calibration_b", 0.0f);
        const auto names_id = gguf_find_key(gguf, "ra.predicate_names");
        if (names_id >= 0) for (std::size_t i = 0; i < gguf_get_arr_n(gguf, names_id); ++i)
            predicate_names.emplace_back(gguf_get_arr_str(gguf, names_id, i));
        if (!options.vocabulary.empty()) {
            const auto bytes = std::filesystem::file_size(options.vocabulary);
            const std::size_t text_dim = meta_u32("ra.text_dim");
            if (!text_dim || !bytes || bytes % (text_dim * sizeof(float)))
                throw std::runtime_error("vocabulary must be a nonempty row-major [V,text_dim] F32 file");
            predicates = bytes / (text_dim * sizeof(float));
            predicate_names.clear();
            vocabulary.resize(bytes / sizeof(float));
            std::ifstream stream(options.vocabulary, std::ios::binary);
            read_exact(stream, vocabulary.data(), vocabulary.size());
            for (std::size_t row = 0; row < predicates; ++row) {
                double norm2 = 0;
                for (std::size_t j = 0; j < text_dim; ++j) {
                    const float v = vocabulary[row * text_dim + j];
                    if (!std::isfinite(v)) throw std::runtime_error("vocabulary contains a nonfinite value");
                    norm2 += v * v;
                }
                const float denom = std::max(1e-12f, static_cast<float>(std::sqrt(norm2)));
                for (std::size_t j = 0; j < text_dim; ++j) vocabulary[row * text_dim + j] /= denom;
            }
        }
        if (predicate_names.empty()) for (std::size_t i = 0; i < predicates; ++i)
            predicate_names.push_back(std::to_string(i));
        if (predicate_names.size() != predicates) throw std::runtime_error("predicate names and weights disagree");
        if (predicates > 131072) throw std::runtime_error("vocabulary exceeds the exact gather-index range");
        // Pack once at model load. Q/K/V have the same input and their output
        // rows are independent, so one matrix product preserves the graph's
        // math while avoiding two dispatches per DINOv3 block.
        packed_qkv.ctx = ggml_init({2 * depth * ggml_tensor_overhead(), nullptr, true});
        if (!packed_qkv.ctx) throw std::runtime_error("cannot create packed attention weights");
        for (std::size_t layer = 0; layer < depth; ++layer) {
            const auto prefix = "ra.backbone.model.model.layer." + std::to_string(layer) + ".attention.";
            auto * weight = t(prefix + "q_proj.weight");
            packed_qkv.weights.push_back(ggml_new_tensor_2d(packed_qkv.ctx, weight->type, c, 3*c));
            packed_qkv.biases.push_back(ggml_new_tensor_1d(packed_qkv.ctx, GGML_TYPE_F32, 3*c));
        }
        packed_qkv.buffer = ggml_backend_alloc_ctx_tensors(packed_qkv.ctx, backend->accelerator);
        if (!packed_qkv.buffer) throw std::runtime_error("cannot allocate packed attention weights");
        ggml_backend_buffer_set_usage(packed_qkv.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (std::size_t layer = 0; layer < depth; ++layer) {
            const auto prefix = "ra.backbone.model.model.layer." + std::to_string(layer) + ".attention.";
            const std::array<std::string, 3> projections = {"q_proj", "k_proj", "v_proj"};
            const auto block_bytes = ggml_nbytes(t(prefix + "q_proj.weight"));
            std::vector<std::uint8_t> bytes(3 * block_bytes);
            std::vector<float> bias(3*c, 0);
            for (std::size_t part = 0; part < 3; ++part) {
                auto * source = t(prefix + projections[part] + ".weight");
                if (source->type != packed_qkv.weights[layer]->type || ggml_nbytes(source) != block_bytes)
                    throw std::runtime_error("incompatible Q/K/V projection layout");
                ggml_backend_tensor_get(source, bytes.data() + part * block_bytes, 0, block_bytes);
                if (part != 1) {
                    auto values = host_f32(t(prefix + projections[part] + ".bias"));
                    std::copy(values.begin(), values.end(), bias.begin() + part * c);
                }
            }
            ggml_backend_tensor_set(packed_qkv.weights[layer], bytes.data(), 0, bytes.size());
            ggml_backend_tensor_set(packed_qkv.biases[layer], bias.data(), 0, bias.size() * sizeof(float));
        }
    }

    ggml_tensor * t(const std::string & n) { return tensor(n); }

    ggml_tensor * affine(ggml_context * ctx, ggml_tensor * x, const std::string & p, bool bias = true) {
        return linear(ctx, x, t("ra." + p + ".weight"), bias ? t("ra." + p + ".bias") : nullptr);
    }

    ggml_tensor * ln(ggml_context * ctx, ggml_tensor * x, const std::string & p) {
        return norm(ctx, x, t("ra." + p + ".weight"), t("ra." + p + ".bias"), 1e-5f);
    }

    ggml_tensor * mha(ggml_context * ctx, ggml_tensor * q, ggml_tensor * kv,
                      const std::string & p, ggml_tensor * mask = nullptr) {
        return packed_attention(ctx, q, kv, kv, t("ra." + p + ".in_proj_weight"),
                                 t("ra." + p + ".in_proj_bias"), t("ra." + p + ".out_proj.weight"),
                                 t("ra." + p + ".out_proj.bias"), q->ne[0], 8, mask,
                                 options.flash_attention && options.backend != "cpu");
    }

    ggml_tensor * encoder(ggml_context * ctx, ggml_tensor * x, const std::string & p,
                          ggml_tensor * mask) {
        auto * y = ln(ctx, x, p + ".norm1");
        x = ggml_add(ctx, x, mha(ctx, y, y, p + ".self_attn", mask));
        y = affine(ctx, ggml_gelu_erf(ctx, affine(ctx, ln(ctx, x, p + ".norm2"), p + ".linear1")), p + ".linear2");
        return ggml_add(ctx, x, y);
    }

    ggml_tensor * decoder(ggml_context * ctx, ggml_tensor * x, ggml_tensor * memory,
                          const std::string & p, ggml_tensor * pair_mask,
                          ggml_tensor * memory_mask, bool last = false) {
        auto * y = ln(ctx, x, p + ".norm1");
        x = ggml_add(ctx, x, mha(ctx, y, y, p + ".self_attn", pair_mask));
        x = ggml_add(ctx, x, mha(ctx, ln(ctx, x, p + ".norm2"), memory,
                                 p + (last ? ".cross_attn" : ".multihead_attn"), memory_mask));
        y = ln(ctx, x, p + ".norm3");
        y = affine(ctx, ggml_gelu_erf(ctx, affine(ctx, y, p + (last ? ".ffn.0" : ".linear1"))),
                    p + (last ? ".ffn.3" : ".linear2"));
        return ggml_add(ctx, x, y);
    }

    ggml_tensor * scene_pe(GraphRun & run, const std::string & p, int h, int w) {
        if (scene_fourier.empty()) {
            std::vector<float> points;
            for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
                points.push_back((x + 0.5f) / w); points.push_back((y + 0.5f) / h);
            }
            scene_fourier = make_fourier_matrix(points, h * w);
        }
        auto * pe = input_tensor(run, run.ctx, GGML_TYPE_F32, 64, h * w, 1, 1, scene_fourier.data());
        return ggml_mul(run.ctx, affine(run.ctx, pe, p + ".proj"), as_f32(run.ctx, t("ra." + p + ".gamma")));
    }

    ggml_tensor * corners(GraphRun & run, const std::vector<float> & boxes,
                           const std::string & p) {
        const std::size_t n = boxes.size() / 4;
        std::vector<float> points;
        std::vector<int32_t> kinds;
        for (std::size_t i = 0; i < n; ++i) {
            auto b = xyxy(boxes.data() + 4 * i);
            points.insert(points.end(), b.begin(), b.end());
            kinds.push_back(0); kinds.push_back(1);
        }
        auto values = make_fourier_matrix(points, 2 * n);
        auto * pe = input_tensor(run, run.ctx, GGML_TYPE_F32, 64, 2 * n, 1, 1, values.data());
        auto * ids = input_tensor(run, run.ctx, GGML_TYPE_I32, 2 * n, 1, 1, 1, kinds.data());
        auto * bias = ggml_get_rows(run.ctx, t("ra." + p + ".corner_bias.weight"), ids);
        return ggml_add(run.ctx, affine(run.ctx, pe, p + ".proj"), as_f32(run.ctx, bias));
    }

    ggml_tensor * pool(GraphRun & run, ggml_tensor * fmap, const std::vector<float> & boxes, int h, int w) {
        auto * ctx = run.ctx;
        const int64_t n = boxes.size() / 4;
        auto * corner = corners(run, boxes, "spatial_pool.box_pe");
        auto * tl = ggml_view_2d(ctx, corner, c, n, 2 * corner->nb[1], 0);
        auto * br = ggml_view_2d(ctx, corner, c, n, 2 * corner->nb[1], corner->nb[1]);
        auto * query = ggml_add(ctx, ggml_scale(ctx, ggml_add(ctx, tl, br), .5f),
                                ggml_reshape_1d(ctx, as_f32(ctx, t("ra.spatial_pool.base_query")), c));
        auto * memory = ggml_add(ctx, fmap, scene_pe(run, "spatial_pool.scene_pe", h, w));
        return ln(ctx, mha(ctx, query, memory, "spatial_pool.cross_attn"), "spatial_pool.norm");
    }

    ggml_tensor * key_mask(GraphRun & run, const std::vector<uint8_t> & valid, int queries) {
        std::vector<float> mask;
        build_mask(mask, valid.size(), queries, valid);
        return input_tensor(run, run.ctx, GGML_TYPE_F32, valid.size(), queries, 1, 1, mask.data());
    }

    ggml_tensor * deformable(GraphRun & run, ggml_tensor * scene, ggml_tensor * offsets,
                             ggml_tensor * weights, const std::vector<float> & anchors,
                             int h, int w, int count) {
        auto * ctx = run.ctx;
        constexpr int points = 16, nh = 8;
        const int dh = d / nh;
        std::vector<float> centers(count * points * 2), half(centers.size());
        for (int i = 0; i < count; ++i) for (int p = 0; p < points; ++p)
            for (int axis = 0; axis < 2; ++axis) {
                centers[(i * points + p) * 2 + axis] = anchors[i * 16 + (p / 4) * 4 + axis];
                half[(i * points + p) * 2 + axis] = .5f * std::max(.05f, anchors[i * 16 + (p / 4) * 4 + axis + 2]);
            }
        auto * cp = input_tensor(run, ctx, GGML_TYPE_F32, 2, points, 1, count, centers.data());
        auto * hp = input_tensor(run, ctx, GGML_TYPE_F32, 2, points, 1, count, half.data());
        auto * pos = ggml_clamp(ctx, ggml_add(ctx,
            ggml_mul(ctx, ggml_reshape_4d(ctx, offsets, 2, points, nh, count), hp), cp), 0, 1);
        // [xy, point, head, pair] -> [point, pair, head, xy]. Keeping the
        // channel vector contiguous makes GET_ROWS a native GPU gather.
        pos = ggml_cont(ctx, ggml_permute(ctx, pos, 3, 0, 2, 1));
        auto * px = ggml_view_3d(ctx, pos, points, count, nh, pos->nb[1], pos->nb[2], 0);
        auto * py = ggml_view_3d(ctx, pos, points, count, nh, pos->nb[1], pos->nb[2], pos->nb[3]);
        px = ggml_scale_bias(ctx, ggml_reshape_2d(ctx, px, points * count, nh), w, .5f);
        py = ggml_scale_bias(ctx, ggml_reshape_2d(ctx, py, points * count, nh), h, .5f);
        auto * x0 = ggml_floor(ctx, px), * y0 = ggml_floor(ctx, py);
        auto * dx = ggml_sub(ctx, px, x0), * dy = ggml_sub(ctx, py, y0);
        auto * base = ggml_add(ctx, x0, ggml_scale(ctx, y0, w + 2));
        auto * grid = ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_4d(ctx, scene, dh, nh, w, h), 0, 3, 1, 2));
        // Zero border reproduces grid_sample(..., padding_mode="zeros",
        // align_corners=False), including half-covered image boundary pixels.
        grid = ggml_pad_ext(ctx, grid, 0, 0, 1, 1, 1, 1, 0, 0);
        grid = ggml_reshape_3d(ctx, grid, dh, (w + 2) * (h + 2), nh);
        ggml_tensor * sampled = nullptr;
        for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x) {
            auto * ids = ggml_cast(ctx, ggml_scale_bias(ctx, base, 1, x + (w + 2) * y), GGML_TYPE_I32);
            auto * wx = x ? dx : ggml_scale_bias(ctx, dx, -1, 1);
            auto * wy = y ? dy : ggml_scale_bias(ctx, dy, -1, 1);
            auto * weighted = ggml_mul(ctx, ggml_get_rows(ctx, grid, ids),
                ggml_reshape_3d(ctx, ggml_mul(ctx, wx, wy), 1, points * count, nh));
            sampled = sampled ? ggml_add(ctx, sampled, weighted) : weighted;
        }
        auto * wh = ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_4d(ctx, weights, 6, 4, nh, count), 0, 1, 3, 2));
        auto * real_weights = ggml_cont(ctx, ggml_view_4d(ctx, wh, 4, 4, count, nh,
            wh->nb[1], wh->nb[2], wh->nb[3], 0));
        sampled = ggml_mul(ctx, sampled, ggml_reshape_3d(ctx, real_weights, 1, points * count, nh));
        sampled = ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_4d(ctx, sampled, dh, points, count, nh), 1, 0, 2, 3));
        auto * read = ggml_reshape_3d(ctx, ggml_sum_rows(ctx, sampled), dh, count, nh);
        auto * null_weights = ggml_cont(ctx, ggml_view_4d(ctx, wh, 2, 4, count, nh,
            wh->nb[1], wh->nb[2], wh->nb[3], 4 * sizeof(float)));
        auto * nulls = ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, as_f32(ctx, t("ra.deformable_read.null_vec")), dh, 8, nh), 1, 0, 2, 3));
        read = ggml_add(ctx, read, matmul(ctx, nulls, ggml_reshape_3d(ctx, null_weights, 8, count, nh)));
        return ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, read, 0, 2, 1, 3)), d, count);
    }

    void dump(const std::string & name, const std::vector<float> & values) const {
        if (options.dump_directory.empty()) return;
        std::filesystem::create_directories(options.dump_directory);
        std::ofstream file(options.dump_directory / (name + ".f32"), std::ios::binary);
        file.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
    }

    std::vector<std::vector<float>> outputs(GraphRun & run,
            const std::vector<std::pair<std::string, ggml_tensor *>> & requested) {
        const auto started = std::chrono::steady_clock::now();
        run.graph = ggml_new_graph_custom(run.ctx, 8192, false);
        for (const auto & item : requested) {
            ggml_set_output(item.second);
            ggml_build_forward_expand(run.graph, item.second);
        }
        execute(run);
        const auto executed = std::chrono::steady_clock::now();
        std::vector<std::vector<float>> values;
        for (const auto & item : requested) { values.push_back(read_f32(item.second)); dump(item.first, values.back()); }
        ggml_backend_sched_reset(backend->scheduler);
        if (options.profile) std::cerr << "outputs read_ms=" << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-executed).count()
            << " total_ms=" << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count() << '\n';
        return values;
    }

    void execute(GraphRun & run) {
        const auto started = std::chrono::steady_clock::now();
        ggml_backend_sched_reset(backend->scheduler);
        for (const auto & item : run.inputs)
            ggml_backend_sched_set_tensor_backend(backend->scheduler, item.first, backend->accelerator);
        if (!ggml_backend_sched_alloc_graph(backend->scheduler, run.graph)) throw std::runtime_error("cannot allocate full graph");
        backend_buffer_bytes = std::max(backend_buffer_bytes,
            ggml_backend_buffer_get_size(weights_buffer) +
            ggml_backend_buffer_get_size(packed_qkv.buffer) +
            ggml_backend_sched_get_buffer_size(backend->scheduler, backend->accelerator));
        const auto allocated = std::chrono::steady_clock::now();
        for (const auto & item : run.inputs) ggml_backend_tensor_set(item.first, item.second.data(), 0, ggml_nbytes(item.first));
        if (ggml_backend_sched_graph_compute(backend->scheduler, run.graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("full graph execution failed");
        ggml_backend_sched_synchronize(backend->scheduler);
        if (options.profile) {
            const auto done = std::chrono::steady_clock::now();
            std::map<std::string, int> cpu_ops;
            for (int i = 0; i < ggml_graph_n_nodes(run.graph); ++i) {
                auto * node = ggml_graph_node(run.graph, i);
                if (ggml_backend_sched_get_tensor_backend(backend->scheduler, node) == backend->cpu)
                    ++cpu_ops[ggml_op_name(node->op)];
            }
            std::cerr << "graph nodes=" << ggml_graph_n_nodes(run.graph)
                      << " allocate_ms=" << std::chrono::duration<double, std::milli>(allocated-started).count()
                      << " compute_ms=" << std::chrono::duration<double, std::milli>(done-allocated).count();
            for (const auto & op : cpu_ops) std::cerr << " cpu_" << op.first << '=' << op.second;
            std::cerr << '\n';
        }
    }

    ggml_tensor * backbone(GraphRun & run, ggml_context * ctx, ggml_tensor * image,
                           std::vector<ggml_tensor *> & taps, int height, int width,
                           ggml_tensor ** raw_patch_out = nullptr) {
        auto * kernel = t("ra.backbone.model.embeddings.patch_embeddings.weight");
        auto * columns = ggml_im2col(ctx, kernel, image, patch, patch, 0, 0, 1, 1, true, GGML_TYPE_F32);
        const int64_t patch_tokens = (height / patch) * (width / patch);
        auto * patches = matmul(ctx, ggml_reshape_2d(ctx, kernel, 3*patch*patch, c),
                                      ggml_reshape_2d(ctx, columns, 3*patch*patch, patch_tokens));
        // GGML [C,N] has the same contiguous storage as PyTorch [N,C].
        patches = ggml_cont(ctx, patches);
        patches = ggml_add(ctx, patches, as_f32(ctx, t("ra.backbone.model.embeddings.patch_embeddings.bias")));
        if (raw_patch_out) *raw_patch_out = patches;
        ggml_tensor * cls = ggml_reshape_2d(ctx, t("ra.backbone.model.embeddings.cls_token"), c, 1);
        ggml_tensor * reg = ggml_reshape_2d(ctx, t("ra.backbone.model.embeddings.register_tokens"), c, regs);
        ggml_tensor * x = concat_axis1(ctx, {cls, reg, patches});
        const int64_t tokens = x->ne[1];
        const int64_t dh = static_cast<int64_t>(c / heads);
        if (rope_height != height || rope_width != width) {
            rope_cos.assign(dh * tokens, 1.0f);
            rope_sin.assign(dh * tokens, 0.0f);
            const float theta = meta_float("ra.rope_theta", 100.0f);
            for (int64_t token = regs + 1; token < tokens; ++token) {
                const int64_t p = token - regs - 1, py = p / (width / patch), px = p % (width / patch);
                for (int64_t j = 0; j < dh; ++j) {
                    const int quarter = static_cast<int>(dh / 4);
                    const int fi = static_cast<int>(j % quarter);
                    const bool y_axis = ((j / quarter) % 2) == 0;
                    const float unit = y_axis ? (static_cast<float>(py) + 0.5f) / (height / patch)
                                              : (static_cast<float>(px) + 0.5f) / (width / patch);
                    // DINOv3 defines patch centers in [-1, 1], not [0, 1].
                    const float coord = 2.0f * unit - 1.0f;
                    const float inv = std::pow(theta,
                                               -4.0f * static_cast<float>(fi) / static_cast<float>(dh));
                    const float angle = 2.0f * static_cast<float>(M_PI) * coord * inv;
                    rope_cos[j + token * dh] = std::cos(angle); rope_sin[j + token * dh] = std::sin(angle);
                }
            }
            rope_height = height;
            rope_width = width;
        }
        ggml_tensor * cos = input_tensor(run, ctx, GGML_TYPE_F32, dh, tokens, 1, 1, rope_cos.data());
        ggml_tensor * sin = input_tensor(run, ctx, GGML_TYPE_F32, dh, tokens, 1, 1, rope_sin.data());
        for (std::size_t layer = 0; layer < depth; ++layer) {
            const std::string pfx = "ra.backbone.model.model.layer." + std::to_string(layer);
            ggml_tensor * y = norm(ctx, x, t(pfx + ".norm1.weight"), t(pfx + ".norm1.bias"), eps);
            auto * qkv = linear(ctx, y, packed_qkv.weights[layer], packed_qkv.biases[layer]);
            ggml_tensor * q = ggml_view_2d(ctx, qkv, c, tokens, qkv->nb[1], 0);
            ggml_tensor * k = ggml_view_2d(ctx, qkv, c, tokens, qkv->nb[1], c * sizeof(float));
            ggml_tensor * v = ggml_view_2d(ctx, qkv, c, tokens, qkv->nb[1], 2 * c * sizeof(float));
            q = ggml_permute(ctx, ggml_reshape_4d(ctx, ggml_cont(ctx, q), dh, heads, tokens, 1), 0, 2, 1, 3);
            k = ggml_permute(ctx, ggml_reshape_4d(ctx, ggml_cont(ctx, k), dh, heads, tokens, 1), 0, 2, 1, 3);
            v = ggml_permute(ctx, ggml_reshape_4d(ctx, ggml_cont(ctx, v), dh, heads, tokens, 1), 0, 2, 1, 3);
            q = rope(ctx, q, cos, sin, dh); k = rope(ctx, k, cos, sin, dh);
            ggml_tensor * attn;
            if (options.flash_attention && options.backend != "cpu") {
                attn = ggml_flash_attn_ext(ctx, ggml_cont(ctx, q),
                    ggml_cast(ctx, k, GGML_TYPE_F16), ggml_cast(ctx, v, GGML_TYPE_F16),
                    nullptr, 1.0f / std::sqrt(static_cast<float>(dh)), 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
            } else {
                ggml_tensor * scores = matmul(ctx, k, q);
                ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / std::sqrt(static_cast<float>(dh)), 0.0f);
                ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
                attn = matmul(ctx, vt, probs);
                attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
            }
            attn = linear(ctx, ggml_reshape_3d(ctx, attn, c, tokens, 1), t(pfx + ".attention.o_proj.weight"), t(pfx + ".attention.o_proj.bias"));
            attn = ggml_mul(ctx, attn, as_f32(ctx, t(pfx + ".layer_scale1.lambda1")));
            x = ggml_add(ctx, x, attn);
            y = norm(ctx, x, t(pfx + ".norm2.weight"), t(pfx + ".norm2.bias"), eps);
            ggml_tensor * mlp;
            if (swiglu) {
                mlp = ggml_mul(ctx, activation(ctx, linear(ctx, y, t(pfx + ".mlp.gate_proj.weight"), t(pfx + ".mlp.gate_proj.bias")), true),
                               linear(ctx, y, t(pfx + ".mlp.up_proj.weight"), t(pfx + ".mlp.up_proj.bias")));
                mlp = linear(ctx, mlp, t(pfx + ".mlp.down_proj.weight"), t(pfx + ".mlp.down_proj.bias"));
            } else {
                mlp = activation(ctx, linear(ctx, y, t(pfx + ".mlp.up_proj.weight"), t(pfx + ".mlp.up_proj.bias")), false);
                mlp = linear(ctx, mlp, t(pfx + ".mlp.down_proj.weight"), t(pfx + ".mlp.down_proj.bias"));
            }
            x = ggml_add(ctx, x, ggml_mul(ctx, mlp, as_f32(ctx, t(pfx + ".layer_scale2.lambda1"))));
            if (layer == 6 || layer == 9 || layer + 1 == depth) taps.push_back(ggml_norm(ctx, ggml_cont(ctx, ggml_view_2d(ctx, x, c, patch_tokens, x->nb[1], static_cast<std::size_t>(regs + 1) * x->nb[1])), 1e-5f));
        }
        auto weights = tap_weights;
        const float mx = *std::max_element(weights.begin(), weights.end());
        float sum = 0;
        for (auto & w : weights) { w = std::exp(w-mx); sum += w; }
        ggml_tensor * fused = ggml_scale(ctx, taps[0], weights[0]/sum);
        for (std::size_t i=1; i<taps.size(); ++i) fused = ggml_add(ctx, fused, ggml_scale(ctx, taps[i], weights[i]/sum));
        return fused;
    }

    ImageOutput infer(const ImageInput & input) {
        if (input.width != 448 || input.height != 448) throw std::runtime_error("full graph currently requires 448x448 input");
        if (input.object_count > options.max_boxes) throw std::runtime_error("object count exceeds Options.max_boxes");
        if (input.rgb_chw.size() != product(3, input.width * input.height, "image") || input.boxes_cxcywh.size() != input.object_count * 4) throw std::runtime_error("invalid RAIM input buffers");
        for (float value : input.rgb_chw) if (!std::isfinite(value)) throw std::runtime_error("image has nonfinite values");
        for (float value : input.boxes_cxcywh) if (!std::isfinite(value)) throw std::runtime_error("boxes have nonfinite values");
        for (std::size_t i = 0; i < input.object_count; ++i)
            if (input.boxes_cxcywh[4*i+2] < 0 || input.boxes_cxcywh[4*i+3] < 0)
                throw std::runtime_error("boxes have negative width or height");
        if (input.object_count < 2) { ImageOutput out; out.predicate_count = predicates; return out; }
        const auto begin = std::chrono::steady_clock::now();
        const std::size_t n = input.object_count;
        const int h = input.height / patch, w = input.width / patch, hw = h * w;
        constexpr int K = 128;
        std::vector<float> patch_data, object_features, geo_scores, rel_scores;
        {
            auto run = make_run();
            auto * image = input_tensor(run, run.ctx, GGML_TYPE_F32, input.width, input.height, 3, 1, input.rgb_chw.data());
            std::vector<ggml_tensor *> taps;
            auto * fmap = backbone(run, run.ctx, image, taps, input.height, input.width);
            auto * objects = pool(run, fmap, input.boxes_cxcywh, h, w);
            std::vector<float> all_geo;
            for (std::size_t s = 0; s < n; ++s) for (std::size_t o = 0; o < n; ++o) {
                auto f = geometry(input.boxes_cxcywh.data() + 4 * s, input.boxes_cxcywh.data() + 4 * o);
                all_geo.insert(all_geo.end(), f.begin(), f.end());
            }
            auto * gf = input_tensor(run, run.ctx, GGML_TYPE_F32, 19, n * n, 1, 1, all_geo.data());
            auto * gs = affine(run.ctx, ggml_relu(run.ctx, affine(run.ctx, gf, "sampler.geo_scorer.0")), "sampler.geo_scorer.2");
            auto scorer = [&](const std::string & p) {
                return affine(run.ctx, ggml_gelu_erf(run.ctx, affine(run.ctx, ln(run.ctx, objects, p + ".0"), p + ".1")), p + ".3");
            };
            auto * zs = scorer("sampler.f_sub"), * zo = scorer("sampler.f_obj");
            // [object,subject] storage is flattened as subject*N + object.
            auto * rs = ggml_scale(run.ctx, matmul(run.ctx, zo, zs), 1.0f / std::sqrt(static_cast<float>(zs->ne[0])));
            auto values = outputs(run, {{"fused_patch_map", fmap}, {"spatial_pool_0", objects}, {"geo_scores", gs}, {"rel_scores", rs}});
            patch_data = std::move(values[0]); object_features = std::move(values[1]);
            geo_scores = std::move(values[2]); rel_scores = std::move(values[3]);
        }
        std::vector<int> candidates(n * n);
        std::iota(candidates.begin(), candidates.end(), 0);
        auto valid_pair = [&](int i) { return i / n != i % n; };
        auto sort_by = [&](const std::vector<float> & scores) {
            std::stable_sort(candidates.begin(), candidates.end(), [&](int a, int b) {
                if (valid_pair(a) != valid_pair(b)) return valid_pair(a);
                return valid_pair(a) && scores[a] > scores[b];
            });
        };
        sort_by(geo_scores); candidates.resize(std::min<std::size_t>(400, candidates.size()));
        sort_by(rel_scores); candidates.resize(std::min<std::size_t>(K, candidates.size()));
        ImageOutput out; out.pair_count = K; out.predicate_count = predicates;
        out.subject_indices.assign(K, 0); out.object_indices.assign(K, 0); out.valid_mask.assign(K, 0);
        out.pair_logits.assign(K, -std::numeric_limits<float>::infinity());
        std::vector<uint8_t> valid(K, 0);
        std::vector<float> sub(K * 4), obj(K * 4), paired_boxes(K * 8), regions(K * 8),
                           anchors(K * 16), geof(K * 19);
        for (int i = 0; i < K; ++i) {
            if (i < static_cast<int>(candidates.size())) {
                int ix = candidates[i]; out.subject_indices[i] = ix / n; out.object_indices[i] = ix % n;
                valid[i] = out.valid_mask[i] = valid_pair(ix); out.pair_logits[i] = rel_scores[ix];
            }
            const float * s = input.boxes_cxcywh.data() + 4 * out.subject_indices[i];
            const float * o = input.boxes_cxcywh.data() + 4 * out.object_indices[i];
            std::copy(s, s + 4, sub.begin() + 4 * i); std::copy(o, o + 4, obj.begin() + 4 * i);
            std::copy(s, s + 4, paired_boxes.begin() + 8 * i); std::copy(o, o + 4, paired_boxes.begin() + 8 * i + 4);
            auto g = geometry(s, o); std::copy(g.begin(), g.end(), geof.begin() + 19 * i);
            float u[4], contact[4];
            const auto sb = xyxy(s), ob = xyxy(o);
            for (int axis = 0; axis < 2; ++axis) {
                const float lo = std::min(s[axis] - .5f * s[axis + 2], o[axis] - .5f * o[axis + 2]);
                const float hi = std::max(s[axis] + .5f * s[axis + 2], o[axis] + .5f * o[axis + 2]);
                u[axis] = .5f * (lo + hi); u[axis + 2] = hi - lo;
                const float inner1 = std::max(sb[axis], ob[axis]), inner2 = std::min(sb[axis + 2], ob[axis + 2]);
                contact[axis] = .5f * (inner1 + inner2); contact[axis + 2] = std::fabs(inner2 - inner1);
            }
            std::copy(u, u + 4, regions.begin() + 4 * i); std::copy(contact, contact + 4, regions.begin() + 4 * (K + i));
            std::copy(s, s + 4, anchors.begin() + 16 * i); std::copy(o, o + 4, anchors.begin() + 16 * i + 4);
            std::copy(u, u + 4, anchors.begin() + 16 * i + 8); std::copy(contact, contact + 4, anchors.begin() + 16 * i + 12);
        }
        dump("subject_indices", std::vector<float>(out.subject_indices.begin(), out.subject_indices.end()));
        dump("object_indices", std::vector<float>(out.object_indices.begin(), out.object_indices.end()));
        dump("valid_mask", std::vector<float>(out.valid_mask.begin(), out.valid_mask.end())); dump("pair_logits", out.pair_logits);
        {
            auto run = make_run(); auto * ctx = run.ctx;
            auto * fmap = input_tensor(run, ctx, GGML_TYPE_F32, c, hw, 1, 1, patch_data.data());
            auto * objects = input_tensor(run, ctx, GGML_TYPE_F32, c, n, 1, 1, object_features.data());
            auto * si = input_tensor(run, ctx, GGML_TYPE_I32, K, 1, 1, 1, out.subject_indices.data());
            auto * oi = input_tensor(run, ctx, GGML_TYPE_I32, K, 1, 1, 1, out.object_indices.data());
            auto * vs = ggml_get_rows(ctx, objects, si), * vo = ggml_get_rows(ctx, objects, oi);
            auto * gp = input_tensor(run, ctx, GGML_TYPE_F32, 19, K, 1, 1, geof.data());
            auto * geo = ln(ctx, affine(ctx, ggml_gelu_erf(ctx, affine(ctx, gp, "geo_encoder.mlp.0")), "geo_encoder.mlp.2"), "geo_encoder.mlp.3");
            auto * bt = corners(run, paired_boxes, "box_prompt_encoder");
            auto * pooled = pool(run, fmap, regions, h, w);
            auto * vu = ggml_view_2d(ctx, pooled, c, K, pooled->nb[1], 0);
            auto * vc = ggml_view_2d(ctx, pooled, c, K, pooled->nb[1], K * pooled->nb[1]);
            auto * pair = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, vs, vo, 0), vu, 0), vc, 0), geo, 0);
            pair = affine(ctx, pair, "pair_proj");
            auto * scene = affine(ctx, fmap, "rel_transformer.scene_proj");
            auto * memory = ggml_concat(ctx, ggml_add(ctx, scene, scene_pe(run, "rel_transformer.scene_pe", h, w)), bt, 1);
            auto * pm = key_mask(run, valid, K);
            std::vector<uint8_t> mv(hw, 1);
            for (int i = 0; i < K; ++i) for (int j = 0; j < 4; ++j) mv.push_back(valid[i]);
            auto * mm = key_mask(run, mv, K);
            auto * rel = encoder(ctx, pair, "rel_transformer.self_layers.0", pm);
            rel = encoder(ctx, rel, "rel_transformer.self_layers.1", pm);
            rel = decoder(ctx, rel, memory, "rel_transformer.cross_layers.0", pm, mm);
            rel = decoder(ctx, rel, memory, "rel_transformer.last_cross", pm, mm, true);
            auto * qn = ln(ctx, rel, "deformable_read.norm");
            auto * offsets = affine(ctx, qn, "deformable_read.offset_mlp");
            auto * weights = affine(ctx, qn, "deformable_read.weight_mlp");
            // The PyTorch view is [K, H, A*(P+S)].  GGML stores the same
            // buffer as [A*(P+S), H, K]; softmax over ne0 therefore performs
            // the per-head/per-pair normalization without a host transpose.
            weights = ggml_soft_max(ctx, ggml_reshape_3d(ctx, weights, 24, 8, K));
            auto * transformer = rel;
            auto * dr = deformable(run, scene, offsets, weights, anchors, h, w, K);
            auto * deform = ggml_add(ctx, rel, ggml_mul(ctx, affine(ctx, dr, "deformable_read.out_proj"), as_f32(ctx, t("ra.deformable_read.gamma"))));
            rel = encoder(ctx, deform, "rel_interaction.dep_layers.0", pm);
            rel = encoder(ctx, rel, "rel_interaction.dep_layers.1", pm);
            memory = ggml_add(ctx, affine(ctx, fmap, "rel_interaction.scene_proj", false), scene_pe(run, "rel_interaction.scene_pe", h, w));
            mv = valid; mv.resize(K + hw, 1);
            rel = encoder(ctx, ggml_concat(ctx, rel, memory, 1), "rel_interaction.gnd_layers.0", key_mask(run, mv, K + hw));
            rel = ggml_cont(ctx, ggml_view_2d(ctx, rel, d, K, rel->nb[1], 0));
            auto * q = affine(ctx, ln(ctx, ggml_gelu_erf(ctx, affine(ctx, rel, "vocab_head.proj.0")), "vocab_head.proj.2"), "vocab_head.proj.3", false);
            const auto & gates = compose_gates;
            q = ln(ctx, ggml_add(ctx, ggml_add(ctx, q, ggml_scale(ctx, affine(ctx, vs, "sub_text_proj", false), gates[0])),
                                  ggml_scale(ctx, affine(ctx, vo, "obj_text_proj", false), gates[1])), "compose_norm");
            auto * qs = affine(ctx, ln(ctx, ggml_gelu_erf(ctx, affine(ctx, ggml_concat(ctx, rel, geo, 0), "spa_proj.0")), "spa_proj.2"), "spa_proj.3", false);
            auto * bank = vocabulary.empty() ? t("ra.vocab_head.W") : input_tensor(run, ctx,
                GGML_TYPE_F32, meta_u32("ra.text_dim"), predicates, 1, 1, vocabulary.data());
            auto * sem = matmul(ctx, bank, ggml_l2_norm(ctx, q, 1e-12f));
            auto * spa = matmul(ctx, bank, ggml_l2_norm(ctx, qs, 1e-12f));
            auto * av = vocabulary.empty() ? as_f32(ctx, t("ra.vocab_head.alpha")) :
                ggml_reshape_1d(ctx, ggml_sigmoid(ctx, affine(ctx, ggml_gelu_erf(ctx,
                    affine(ctx, bank, "vocab_head.gate_mlp.0")), "vocab_head.gate_mlp.2")), predicates);
            auto * ia = ggml_scale_bias(ctx, av, -1, 1);
            auto * logits = ggml_add(ctx, ggml_mul(ctx, sem, ia), ggml_mul(ctx, spa, av));
            const float scale = vocabulary_scale;
            logits = ggml_add(ctx, ggml_scale(ctx, logits, scale), as_f32(ctx, t("ra.vocab_head.logit_bias")));
            auto * best_id = ggml_cast(ctx, ggml_argmax(ctx, logits), GGML_TYPE_F32);
            std::vector<float> row_offsets(K);
            for (int i = 0; i < K; ++i) row_offsets[i] = static_cast<float>(i * predicates);
            auto * base = input_tensor(run, ctx, GGML_TYPE_F32, K, 1, 1, 1, row_offsets.data());
            auto * indices = ggml_cast(ctx, ggml_add(ctx, best_id, base), GGML_TYPE_I32);
            auto * best_logit = ggml_get_rows(ctx, ggml_reshape_2d(ctx, logits, 1, K * predicates), indices);
            std::vector<std::pair<std::string, ggml_tensor *>> requested = {
                {"best_predicate", best_id}, {"best_logit", best_logit}};
            if (options.raw_logits || !options.dump_directory.empty())
                requested.emplace_back("predicate_logits", logits);
            if (!options.dump_directory.empty()) {
                requested.insert(requested.end(), {{"deformable_read", deform}, {"rel_interaction", rel},
                    {"vocab_query", q}, {"spa_proj", qs}, {"vocab_semantic", sem}, {"vocab_spatial", spa},
                    {"rel_transformer", transformer}, {"scene_projected", scene}, {"deformable_offsets", offsets},
                    {"deformable_weights", weights}, {"geo_encoder", geo}, {"vsub", vs}, {"vobj", vo},
                    {"box_prompt_tokens", bt}, {"spatial_pool_1", pooled}, {"pair_proj", pair}, {"deformable_sample", dr}});
            }
            auto values = outputs(run, requested);
            for (int i = 0; i < K; ++i) if (out.valid_mask[i]) {
                const float z = calibration_a * (values[1][i] + out.pair_logits[i]) + calibration_b;
                out.relations.push_back({static_cast<std::size_t>(out.subject_indices[i]),
                    static_cast<std::size_t>(out.object_indices[i]), static_cast<std::size_t>(values[0][i]),
                    1.0f / (1.0f + std::exp(-std::clamp(z, -80.0f, 80.0f)))});
            }
            std::stable_sort(out.relations.begin(), out.relations.end(),
                [](const Relation & a, const Relation & b) { return a.score > b.score; });
            if (options.raw_logits || !options.dump_directory.empty())
                out.predicate_logits = std::move(values[2]);
        }
        out.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        out.backend_buffer_bytes = backend_buffer_bytes;
        return out;
    }
};

FullModel::FullModel(const Options & options) : impl_(std::make_unique<Impl>(options)) {}
FullModel::~FullModel() = default;
ImageOutput FullModel::infer(const ImageInput & input) { return impl_->infer(input); }
const std::vector<std::string> & FullModel::predicates() const { return impl_->predicate_names; }

bool is_image_input(const std::filesystem::path & path) {
    return path.extension() == ".raim" || path.extension() == ".raip";
}

ImageInput read_image_input(const std::filesystem::path & path) {
    if (path.extension() == ".raip") return read_image_request(path);
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open RAIM input: " + path.string());
    char magic[sizeof(kImageMagic)]{}; read_exact(stream, magic, sizeof(magic));
    if (std::memcmp(magic, kImageMagic, sizeof(magic)) != 0) throw std::runtime_error("invalid RAIMv1 magic");
    std::uint32_t width = 0, height = 0, objects = 0;
    read_exact(stream, &width); read_exact(stream, &height); read_exact(stream, &objects);
    if (width != 448 || height != 448 || objects > 10000)
        throw std::runtime_error("invalid RAIM dimensions");
    const std::uintmax_t expected = 20 + (std::uintmax_t(width) * height * 3 + std::uintmax_t(objects) * 4) * 4;
    if (std::filesystem::file_size(path) != expected) throw std::runtime_error("invalid RAIM byte count");
    ImageInput input; input.width = width; input.height = height; input.object_count = objects;
    input.rgb_chw.resize(product(3, product(width, height, "RAIM image"), "RAIM image"));
    input.boxes_cxcywh.resize(product(objects, std::size_t(4), "RAIM boxes"));
    read_exact(stream, input.rgb_chw.data(), input.rgb_chw.size()); read_exact(stream, input.boxes_cxcywh.data(), input.boxes_cxcywh.size());
    return input;
}

void write_image_output(const std::filesystem::path & path, const ImageOutput & output) {
    if (output.predicate_logits.size() != output.pair_count * output.predicate_count || output.valid_mask.size() != output.pair_count) throw std::runtime_error("invalid full graph output");
    std::ofstream stream(path, std::ios::binary); if (!stream) throw std::runtime_error("cannot create output: " + path.string());
    stream.write(kImageOutputMagic, sizeof(kImageOutputMagic));
    const auto p = static_cast<std::uint32_t>(output.pair_count), v = static_cast<std::uint32_t>(output.predicate_count);
    stream.write(reinterpret_cast<const char *>(&p), sizeof(p)); stream.write(reinterpret_cast<const char *>(&v), sizeof(v));
    stream.write(reinterpret_cast<const char *>(output.valid_mask.data()), static_cast<std::streamsize>(output.valid_mask.size() * sizeof(std::int32_t)));
    stream.write(reinterpret_cast<const char *>(output.subject_indices.data()), static_cast<std::streamsize>(output.subject_indices.size() * sizeof(std::int32_t)));
    stream.write(reinterpret_cast<const char *>(output.object_indices.data()), static_cast<std::streamsize>(output.object_indices.size() * sizeof(std::int32_t)));
    stream.write(reinterpret_cast<const char *>(output.pair_logits.data()), static_cast<std::streamsize>(output.pair_logits.size() * sizeof(float)));
    stream.write(reinterpret_cast<const char *>(output.predicate_logits.data()), static_cast<std::streamsize>(output.predicate_logits.size() * sizeof(float)));
    if (!stream) throw std::runtime_error("failed writing full graph output");
}

void write_image_input(const std::filesystem::path & path, const ImageInput & input) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot write RAIM: " + path.string());
    stream.write(kImageMagic, sizeof(kImageMagic));
    const std::uint32_t dims[] = {static_cast<std::uint32_t>(input.width),
        static_cast<std::uint32_t>(input.height), static_cast<std::uint32_t>(input.object_count)};
    stream.write(reinterpret_cast<const char *>(dims), sizeof(dims));
    stream.write(reinterpret_cast<const char *>(input.rgb_chw.data()), input.rgb_chw.size() * sizeof(float));
    stream.write(reinterpret_cast<const char *>(input.boxes_cxcywh.data()), input.boxes_cxcywh.size() * sizeof(float));
    if (!stream) throw std::runtime_error("failed writing RAIM");
}

} // namespace relateanything
