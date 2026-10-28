#pragma once

#include "relateanything/runtime_options.hpp"
#include "relateanything/backend_binding.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace relateanything {

struct ModelOptions {
    RuntimeOptions runtime;
    ra_backend_binding backend_binding{};
    std::size_t device_index = 0;
    std::filesystem::path model;
    std::filesystem::path dump_directory;
    std::filesystem::path vocabulary;
    std::filesystem::path vocabulary_names;
    std::filesystem::path predicate_type_map;
    std::filesystem::path text_student;
    std::filesystem::path tokenizer;
    std::filesystem::path texts;
    std::filesystem::path token_output;
    std::vector<std::string> text_templates;
    std::string backend = "cpu";
    std::size_t threads = 0;
    std::size_t max_boxes = 60;
    std::size_t topk = 128;
    bool decompose = false;
    bool flash_attention = true;
    bool compact_pairs = true;
    bool matmul_sampler = false;
    bool matmul_geometry = false;
    bool fused_vocabulary = true;
    bool cache_positions = true;
    bool retain_features = true;
    bool quant_f32_input = false;
    bool strict_f32 = false;
    // Replace strict_f32 after each GGUF is read: true on CUDA when every
    // matrix weight is F32, false otherwise. This is the default for the
    // executable and the C API; --strict-f32 / --no-strict-f32 and a
    // non-negative ra_options.strict_f32 clear it to force a fixed mode.
    bool strict_f32_auto = true;
    bool raw_logits = false;
    bool profile = false;

    void validate_model() const;
};

// File I/O and benchmark controls belong to the executable, not the session.
struct Options : ModelOptions {
    std::filesystem::path input;
    std::filesystem::path output;
    std::size_t warmup = 2;
    std::size_t repeats = 10;
    bool encode_text_only = false;
    bool preprocess_only = false;
    bool self_test = false;

    static Options from_args(int argc, char ** argv);
    static void print_help(const char * program);
    void validate() const;
};

} // namespace relateanything
