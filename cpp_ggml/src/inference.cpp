#include "relateanything/options.hpp"

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace relateanything {

namespace {

std::string require_value(int argc, char ** argv, int & index, const char * option) {
    if (index + 1 >= argc) throw std::invalid_argument(std::string(option) + " needs a value");
    return argv[++index];
}

std::size_t parse_size(const std::string & value, const char * option) {
    std::size_t result = 0;
    const auto * begin = value.data();
    const auto * end = begin + value.size();
    const auto [ptr, error] = std::from_chars(begin, end, result);
    if (error != std::errc() || ptr != end) {
        throw std::invalid_argument(std::string(option) + " must be a non-negative integer");
    }
    return result;
}

} // namespace

Options Options::from_args(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else if (arg == "--self-test") {
            options.self_test = true;
        } else if (arg == "--model") {
            options.model = require_value(argc, argv, i, "--model");
        } else if (arg == "--input") {
            options.input = require_value(argc, argv, i, "--input");
        } else if (arg == "--output") {
            options.output = require_value(argc, argv, i, "--output");
        } else if (arg == "--dump-directory") {
            options.dump_directory = require_value(argc, argv, i, "--dump-directory");
        } else if (arg == "--vocabulary") {
            options.vocabulary = require_value(argc, argv, i, "--vocabulary");
        } else if (arg == "--flash-attention") {
            options.flash_attention = true;
        } else if (arg == "--no-flash-attention") {
            options.flash_attention = false;
        } else if (arg == "--raw-logits") {
            options.raw_logits = true;
        } else if (arg == "--preprocess-only") {
            options.preprocess_only = true;
        } else if (arg == "--backend") {
            options.backend = require_value(argc, argv, i, "--backend");
        } else if (arg == "--threads") {
            options.threads = parse_size(require_value(argc, argv, i, "--threads"), "--threads");
        } else if (arg == "--warmup") {
            options.warmup = parse_size(require_value(argc, argv, i, "--warmup"), "--warmup");
        } else if (arg == "--repeats") {
            options.repeats = parse_size(require_value(argc, argv, i, "--repeats"), "--repeats");
        } else if (arg == "--max-boxes") {
            options.max_boxes = parse_size(require_value(argc, argv, i, "--max-boxes"), "--max-boxes");
        } else if (arg == "--profile") {
            options.profile = true;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    options.validate();
    return options;
}

void Options::validate() const {
    if (backend != "cpu" && backend != "cuda" && backend != "vulkan") {
        throw std::invalid_argument("--backend must be cpu, cuda or vulkan");
    }
    if (repeats == 0) throw std::invalid_argument("--repeats must be greater than zero");
    if (max_boxes == 0) throw std::invalid_argument("--max-boxes must be greater than zero");
    if (!self_test && !preprocess_only && model.empty()) throw std::invalid_argument("--model is required");
    if (preprocess_only && output.empty()) throw std::invalid_argument("--preprocess-only requires --output");
    if (!self_test && input.empty()) throw std::invalid_argument("--input is required");
}

void Options::print_help(const char * program) {
    std::printf("Usage: %s --model MODEL.gguf --input INPUT.raim|DIRECTORY [options]\n", program);
    std::printf("  --backend cpu|cuda|vulkan   execution backend (default cpu)\n");
    std::printf("  --output PATH               write calibrated predictions (JSON)\n");
    std::printf("  --raw-logits                write RAFO tensors for numerical comparison\n");
    std::printf("  --preprocess-only           convert image requests (.raip) to normalized RAIM\n");
    std::printf("  --dump-directory PATH       write intermediate F32 tensors\n");
    std::printf("  --vocabulary PATH           replace vocabulary with [V,512] F32 embeddings\n");
    std::printf("  --flash-attention           use backend flash attention\n");
    std::printf("  --no-flash-attention        use explicit F32 attention\n");
    std::printf("  --max-boxes N               maximum objects (default 60)\n");
    std::printf("  --threads N                 CPU worker count (0 = ggml default)\n");
    std::printf("  --warmup N --repeats N     benchmark controls\n");
    std::printf("  --profile                   print timing and tensor details\n");
    std::printf("  --self-test                 run a backend-independent graph test\n");
}

} // namespace relateanything
