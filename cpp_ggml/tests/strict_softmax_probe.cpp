#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

// An independent Python CUDA oracle supplies the input and expected output.
// This executable measures the actual GGML operation, including dispatch.
int main(int argc, char ** argv) {
    if (argc != 8) {
        std::cerr << "usage: probe input output columns rows scale strict repeats\n";
        return 2;
    }
    try {
        const int cols = std::stoi(argv[3]), rows = std::stoi(argv[4]);
        const float scale = std::stof(argv[5]);
        const bool strict = std::stoi(argv[6]) != 0;
        const int repeats = std::stoi(argv[7]);
        if (cols < 1 || rows < 1 || repeats < 1) throw std::runtime_error("invalid shape/repeats");
        std::vector<float> values(size_t(cols)*rows);
        std::ifstream input(argv[1], std::ios::binary);
        input.read(reinterpret_cast<char *>(values.data()), values.size()*sizeof(float));
        if (!input || input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("invalid input size");
        auto * backend = ggml_backend_cuda_init(0);
        if (!backend) throw std::runtime_error("CUDA backend unavailable");
        auto * ctx = ggml_init({ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true});
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        auto * y = ggml_soft_max_ext(ctx, x, nullptr, scale, 0.0f);
        if (strict) ggml_soft_max_set_hint(y, GGML_HINT_STRICT_F32);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("allocation failed");
        ggml_backend_tensor_set(x, values.data(), 0, values.size()*sizeof(float));
        auto run = [&] {
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("graph failed");
            ggml_backend_synchronize(backend);
        };
        for (int i = 0; i < 5; ++i) run();
        std::vector<double> timings;
        for (int i = 0; i < repeats; ++i) {
            auto started = std::chrono::steady_clock::now();
            run();
            timings.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now()-started).count());
        }
        ggml_backend_tensor_get(y, values.data(), 0, values.size()*sizeof(float));
        std::ofstream output(argv[2], std::ios::binary);
        output.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float));
        if (!output) throw std::runtime_error("output write failed");
        std::sort(timings.begin(), timings.end());
        std::cout << "median_ms=" << timings[timings.size()/2] << '\n';
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
