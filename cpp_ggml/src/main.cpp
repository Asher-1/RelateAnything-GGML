#include "relateanything/backend.hpp"
#include "relateanything/gguf_model.hpp"
#include "relateanything/full_model.hpp"
#include "relateanything/options.hpp"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace relateanything {

namespace {

std::string json_string(const std::string & text) {
    std::string out = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char ch : text) {
        if (ch == '"' || ch == '\\') { out += '\\'; out += static_cast<char>(ch); }
        else if (ch < 32) { out += "\\u00"; out += hex[ch >> 4]; out += hex[ch & 15]; }
        else out += static_cast<char>(ch);
    }
    return out + '"';
}

void write_predictions(const std::filesystem::path & path, const relateanything::ImageOutput & output,
                       const std::vector<std::string> & names) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("cannot create prediction JSON: " + path.string());
    stream << std::setprecision(9) << "{\"latency_ms\":" << output.milliseconds
           << ",\"backend_buffer_bytes\":" << output.backend_buffer_bytes << ",\"predictions\":[";
    bool first = true;
    for (const auto & r : output.relations) {
        if (!first) stream << ',';
        first = false;
        stream << "{\"subject\":" << r.subject << ",\"object\":" << r.object
               << ",\"predicate_id\":" << r.predicate << ",\"predicate\":" << json_string(names.at(r.predicate))
               << ",\"score\":" << r.score << '}';
    }
    stream << "]}\n";
    if (!stream) throw std::runtime_error("failed writing prediction JSON");
}

int run_self_test(const Options & options) {
    Backend backend(options);
    constexpr int rows_a = 4;
    constexpr int cols_a = 2;
    constexpr int rows_b = 3;
    constexpr int cols_b = 2;
    const std::array<float, rows_a * cols_a> matrix_a = {2, 8, 5, 1, 4, 2, 8, 6};
    const std::array<float, rows_b * cols_b> matrix_b = {10, 5, 9, 9, 5, 4};
    const std::size_t bytes = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    ggml_init_params params{bytes, nullptr, true};
    ggml_context * context = ggml_init(params);
    if (!context) throw std::runtime_error("cannot create self-test context");
    ggml_tensor * a = ggml_new_tensor_2d(context, GGML_TYPE_F32, cols_a, rows_a);
    ggml_tensor * b = ggml_new_tensor_2d(context, GGML_TYPE_F32, cols_b, rows_b);
    ggml_set_input(a);
    ggml_set_input(b);
    ggml_tensor * result = ggml_mul_mat(context, a, b);
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, result);
    ggml_backend_sched_reset(backend.scheduler);
    if (!ggml_backend_sched_alloc_graph(backend.scheduler, graph)) {
        ggml_free(context);
        throw std::runtime_error("self-test graph allocation failed");
    }
    ggml_backend_tensor_set(a, matrix_a.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, matrix_b.data(), 0, ggml_nbytes(b));
    if (ggml_backend_sched_graph_compute(backend.scheduler, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(context);
        throw std::runtime_error("self-test graph execution failed");
    }
    std::vector<float> actual(ggml_nelements(result));
    ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
    const std::array<float, 12> expected = {60, 55, 50, 110, 90, 54, 54, 126, 42, 29, 28, 64};
    float max_error = 0.0f;
    for (std::size_t i = 0; i < expected.size(); ++i) max_error = std::max(max_error, std::fabs(actual[i] - expected[i]));
    ggml_free(context);
    if (max_error > 1e-4f) throw std::runtime_error("self-test numerical mismatch");
    std::cout << "SELF_TEST PASS backend=" << options.backend << " max_abs=" << max_error << '\n';
    return 0;
}

} // namespace

} // namespace relateanything

int main(int argc, char ** argv) {
    try {
        const relateanything::Options options = relateanything::Options::from_args(argc, argv);
        if (options.self_test) return relateanything::run_self_test(options);
        if (relateanything::is_image_input(options.input) || std::filesystem::is_directory(options.input)) {
            std::vector<std::filesystem::path> paths;
            const bool batch = std::filesystem::is_directory(options.input);
            if (batch) {
                for (const auto & entry : std::filesystem::directory_iterator(options.input))
                    if (entry.is_regular_file() && relateanything::is_image_input(entry.path())) paths.push_back(entry.path());
                std::sort(paths.begin(), paths.end());
                if (paths.empty()) throw std::runtime_error("input directory contains no .raim files");
                if (!options.output.empty()) std::filesystem::create_directories(options.output);
            } else paths.push_back(options.input);
            if (options.preprocess_only) {
                for (const auto & path : paths) relateanything::write_image_input(
                    batch ? options.output / (path.stem().string() + ".raim") : options.output,
                    relateanything::read_image_input(path));
                return 0;
            }
            relateanything::FullModel model(options);
            bool warmed = false;
            for (const auto & path : paths) {
                const auto input = relateanything::read_image_input(path);
                if (!warmed) for (std::size_t i = 0; i < options.warmup; ++i) model.infer(input);
                warmed = true;
                relateanything::ImageOutput output;
                std::vector<double> times;
                double total = 0.0;
                for (std::size_t i = 0; i < options.repeats; ++i) {
                    const auto start = std::chrono::steady_clock::now();
                    output = model.infer(path.extension() == ".raip" ? relateanything::read_image_input(path) : input);
                    output.milliseconds = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count();
                    times.push_back(output.milliseconds);
                    total += output.milliseconds;
                }
                output.iteration_milliseconds = times;
                output.milliseconds = total / static_cast<double>(options.repeats);
                if (!options.output.empty()) {
                    const auto dest = batch ? options.output / (path.stem().string() +
                        (options.raw_logits ? ".rafo" : ".json")) : options.output;
                    if (options.raw_logits) relateanything::write_image_output(dest, output);
                    else relateanything::write_predictions(dest, output, model.predicates());
                }
                std::cout << "sample=" << path.stem().string() << '\n'
                          << "backend=" << options.backend << '\n'
                          << "architecture=relateanything_full_v1\n"
                          << "pairs=" << output.pair_count << '\n'
                          << "predicates=" << output.predicate_count << '\n'
                          << "latency_ms=" << output.milliseconds << '\n'
                          << "iteration_ms=";
                for (std::size_t i = 0; i < times.size(); ++i) {
                    if (i) std::cout << ',';
                    std::cout << times[i];
                }
                std::cout << '\n';
            }
            return 0;
        }
        relateanything::Model model(options);
        const relateanything::PairInput input = relateanything::read_pair_input(options.input);
        const relateanything::PairOutput output = model.infer(input);
        if (!options.output.empty()) relateanything::write_pair_output(options.output, output);
        std::cout << "backend=" << options.backend << '\n'
                  << "architecture=" << model.architecture() << '\n'
                  << "pairs=" << output.pair_count << '\n'
                  << "predicates=" << output.predicate_count << '\n'
                  << "latency_ms=" << output.milliseconds << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "relateanything-ggml: " << error.what() << '\n';
        return 2;
    }
}
