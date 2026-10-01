#pragma once

#include "relateanything/backend.hpp"

#include "ggml.h"
#include "gguf.h"

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace relateanything {

struct PairInput {
    std::size_t object_count = 0;
    std::size_t feature_dim = 0;
    std::size_t feature_stride = 0;
    std::vector<float> object_features;
    std::vector<float> boxes_cxcywh;
};

struct PairOutput {
    std::size_t pair_count = 0;
    std::size_t predicate_count = 0;
    std::vector<float> predicate_logits;
    std::vector<float> pair_logits;
    std::vector<std::int32_t> subject_indices;
    std::vector<std::int32_t> object_indices;
    double milliseconds = 0.0;
};

class Model {
public:
    explicit Model(const Options & options);
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;
    ~Model();

    PairOutput infer(const PairInput & input);
    const std::string & architecture() const { return architecture_; }
    std::size_t predicate_count() const { return predicate_count_; }

private:
    Options options_;
    gguf_context * gguf_ = nullptr;
    ggml_context * tensor_context_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    std::unique_ptr<Backend> backend_;
    std::map<std::string, ggml_tensor *> tensors_;
    std::string architecture_;
    std::size_t feature_dim_ = 0;
    std::size_t hidden_dim_ = 0;
    std::size_t predicate_count_ = 0;

    ggml_tensor * tensor(const char * name) const;
    static void require(bool condition, const std::string & message);
};

PairInput read_pair_input(const std::filesystem::path & path);
void write_pair_output(const std::filesystem::path & path, const PairOutput & output);

} // namespace relateanything
