#pragma once

#include "relateanything/gguf_model.hpp"

namespace relateanything {

struct ImageOptions {
    std::filesystem::path image;
    std::vector<float> boxes_xyxy;
};

struct ImageInput {
    std::size_t width = 448;
    std::size_t height = 448;
    std::size_t object_count = 0;
    // RGB, CHW, ImageNet normalized after Pillow-compatible bilinear resize.
    std::vector<float> rgb_chw;
    std::vector<float> boxes_cxcywh;
};

struct Relation {
    std::size_t subject = 0;
    std::size_t object = 0;
    std::size_t predicate = 0;
    float score = 0;
};

struct ImageOutput : PairOutput {
    std::vector<std::int32_t> valid_mask;
    std::vector<double> iteration_milliseconds;
    std::vector<Relation> relations;
    // GGML-owned accelerator weight/graph buffers; excludes driver allocations.
    std::size_t backend_buffer_bytes = 0;
};

class FullModel {
public:
    explicit FullModel(const Options & options);
    ~FullModel();
    FullModel(const FullModel &) = delete;
    FullModel & operator=(const FullModel &) = delete;
    ImageOutput infer(const ImageInput & input);
    const std::vector<std::string> & predicates() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool is_image_input(const std::filesystem::path & path);
ImageInput prepare_image(const ImageOptions & options);
ImageInput read_image_request(const std::filesystem::path & path);
ImageInput read_image_input(const std::filesystem::path & path);
void write_image_input(const std::filesystem::path & path, const ImageInput & input);
void write_image_output(const std::filesystem::path & path, const ImageOutput & output);

} // namespace relateanything
