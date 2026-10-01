// Pillow-compatible fixed-point resize. The coefficient construction follows
// Pillow's HPND-licensed Resample.c; see third_party/patches/README.md.
// Local precedent: General-Keypoint-Detection-GGML/cpp_ggml/src/image_io.cpp.
#include "relateanything/full_model.hpp"
#include <algorithm>
#include <cmath>
#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <jpeglib.h>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

namespace relateanything {
namespace {
struct Pixels {
    int width = 0, height = 0;
    std::vector<unsigned char> rgb;
};
struct JpegError { jpeg_error_mgr base; jmp_buf jump; };
void jpeg_failure(j_common_ptr info) { longjmp(reinterpret_cast<JpegError *>(info->err)->jump, 1); }
Pixels load_rgb(const std::filesystem::path & path) {
    FILE * stream = std::fopen(path.string().c_str(), "rb");
    if (!stream) throw std::runtime_error("cannot open image: " + path.string());
    const int a = std::fgetc(stream), b = std::fgetc(stream);
    std::rewind(stream);
    Pixels pixels;
    if (a == 0xff && b == 0xd8) {
        // Heap state survives libjpeg's error longjmp; no C++ objects are
        // created between setjmp and any call that can invoke error_exit.
        auto * info = new jpeg_decompress_struct{};
        JpegError error{};
        info->err = jpeg_std_error(&error.base); error.base.error_exit = jpeg_failure;
        if (setjmp(error.jump)) {
            jpeg_destroy_decompress(info); delete info; std::fclose(stream);
            throw std::runtime_error("invalid JPEG: " + path.string());
        }
        jpeg_create_decompress(info); jpeg_stdio_src(info, stream); jpeg_read_header(info, TRUE);
        info->out_color_space = JCS_RGB;
        jpeg_start_decompress(info);
        if (!info->output_width || !info->output_height || info->output_width > 20000 || info->output_height > 20000) {
            jpeg_destroy_decompress(info); delete info; std::fclose(stream);
            throw std::runtime_error("unsupported image dimensions");
        }
        pixels.width = info->output_width; pixels.height = info->output_height;
        pixels.rgb.resize(static_cast<std::size_t>(pixels.width) * pixels.height * 3);
        while (info->output_scanline < info->output_height) {
            JSAMPROW row = pixels.rgb.data() + static_cast<std::size_t>(info->output_scanline) * pixels.width * 3;
            jpeg_read_scanlines(info, &row, 1);
        }
        jpeg_finish_decompress(info); jpeg_destroy_decompress(info); delete info; std::fclose(stream);
    } else {
        std::fclose(stream);
        int channels;
        auto * raw = stbi_load(path.string().c_str(), &pixels.width, &pixels.height, &channels, 3);
        if (!raw) throw std::runtime_error("cannot decode image: " + path.string());
        pixels.rgb.assign(raw, raw + static_cast<std::size_t>(pixels.width) * pixels.height * 3);
        stbi_image_free(raw);
    }
    return pixels;
}
struct Axis {
    std::vector<int> start;
    std::vector<std::vector<int>> weights;
};
Axis axis(int src, int dst) {
    Axis out;
    const double scale = double(src) / dst, support = std::max(scale, 1.0);
    for (int i = 0; i < dst; ++i) {
        const double center = (i + .5) * scale;
        const int lo = std::max(0, int(center - support + .5));
        const int hi = std::min(src, int(center + support + .5));
        std::vector<double> raw; double sum = 0;
        for (int j = lo; j < hi; ++j) {
            const double value = std::max(0.0, 1.0 - std::abs((j - center + .5) / support));
            raw.push_back(value); sum += value;
        }
        std::vector<int> weights;
        for (double value : raw) weights.push_back(int(.5 + value / sum * (1 << 22)));
        out.start.push_back(lo); out.weights.push_back(std::move(weights));
    }
    return out;
}
Pixels resize(const Pixels & src) {
    constexpr int size = 448;
    const Axis xs = axis(src.width, size), ys = axis(src.height, size);
    std::vector<unsigned char> tmp(static_cast<std::size_t>(size) * src.height * 3);
    for (int y = 0; y < src.height; ++y) for (int x = 0; x < size; ++x) {
        int a0 = 1 << 21, a1 = a0, a2 = a0;
        for (std::size_t j = 0; j < xs.weights[x].size(); ++j) {
            const auto * pixel = src.rgb.data() + (static_cast<std::size_t>(y) * src.width + xs.start[x] + j) * 3;
            const int weight = xs.weights[x][j];
            a0 += pixel[0] * weight; a1 += pixel[1] * weight; a2 += pixel[2] * weight;
        }
        auto * pixel = tmp.data() + (static_cast<std::size_t>(y) * size + x) * 3;
        pixel[0] = std::clamp(a0 >> 22, 0, 255);
        pixel[1] = std::clamp(a1 >> 22, 0, 255);
        pixel[2] = std::clamp(a2 >> 22, 0, 255);
    }
    Pixels dst{size, size, std::vector<unsigned char>(size * size * 3)};
    std::vector<int> row(size * 3);
    for (int y = 0; y < size; ++y) {
        std::fill(row.begin(), row.end(), 1 << 21);
        for (std::size_t j = 0; j < ys.weights[y].size(); ++j) {
            const auto * source = tmp.data() + (ys.start[y] + j) * size * 3;
            const int weight = ys.weights[y][j];
            // Contiguous row arithmetic allows compiler SIMD while preserving
            // the fixed-point summation and rounding of the PIL reference.
            for (int x = 0; x < size * 3; ++x) row[x] += source[x] * weight;
        }
        auto * target = dst.rgb.data() + y * size * 3;
        for (int x = 0; x < size * 3; ++x)
            target[x] = std::clamp(row[x] >> 22, 0, 255);
    }
    return dst;
}
} // namespace

ImageInput prepare_image(const ImageOptions & options) {
    const auto original = load_rgb(options.image);
    const auto pixels = resize(original);
    ImageInput input;
    input.rgb_chw.resize(3 * 448 * 448);
    constexpr float mean[] = {.485f, .456f, .406f}, deviation[] = {.229f, .224f, .225f};
    for (int c = 0; c < 3; ++c) for (int i = 0; i < 448 * 448; ++i)
        input.rgb_chw[c * 448 * 448 + i] = (pixels.rgb[3 * i + c] / 255.0f - mean[c]) / deviation[c];
    if (options.boxes_xyxy.size() % 4) throw std::runtime_error("boxes must contain four coordinates each");
    input.object_count = options.boxes_xyxy.size() / 4;
    for (std::size_t i = 0; i < input.object_count; ++i) {
        const float * b = options.boxes_xyxy.data() + i * 4;
        // Match upstream operation order: normalize corners before converting
        // to cxcywh. A one-ULP geometry difference is amplified by Fourier PE.
        const float x1 = b[0] / original.width, y1 = b[1] / original.height;
        const float x2 = b[2] / original.width, y2 = b[3] / original.height;
        input.boxes_cxcywh.insert(input.boxes_cxcywh.end(), {
            (x1 + x2) * .5f, (y1 + y2) * .5f, x2 - x1, y2 - y1});
    }
    return input;
}

ImageInput read_image_request(const std::filesystem::path & path) {
    std::ifstream stream(path);
    std::string magic, image;
    std::size_t count = 0;
    stream >> magic >> std::quoted(image) >> count;
    if (!stream || magic != "RAIP1" || count > 10000) throw std::runtime_error("invalid RAIP request");
    ImageOptions options;
    options.image = std::filesystem::path(image);
    if (options.image.is_relative()) options.image = path.parent_path() / options.image;
    options.boxes_xyxy.resize(count * 4);
    // Python JSON numbers are parsed as double before np.float32 conversion.
    // Match that rounding for decimal coordinates exactly on a float midpoint.
    for (float & value : options.boxes_xyxy) {
        double parsed = 0;
        stream >> parsed;
        value = static_cast<float>(parsed);
    }
    if (!stream) throw std::runtime_error("truncated RAIP boxes");
    return prepare_image(options);
}
} // namespace relateanything
