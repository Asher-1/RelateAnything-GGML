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
        } else if (arg == "--vocabulary-names") {
            options.vocabulary_names = require_value(argc, argv, i, "--vocabulary-names");
        } else if (arg == "--predicate-type-map") {
            options.predicate_type_map = require_value(argc, argv, i, "--predicate-type-map");
        } else if (arg == "--decompose") {
            options.decompose = true;
        } else if (arg == "--topk") {
            options.topk = parse_size(require_value(argc, argv, i, "--topk"), "--topk");
        } else if (arg == "--text-student") {
            options.text_student = require_value(argc, argv, i, "--text-student");
        } else if (arg == "--tokenizer") {
            options.tokenizer = require_value(argc, argv, i, "--tokenizer");
        } else if (arg == "--texts") {
            options.texts = require_value(argc, argv, i, "--texts");
        } else if (arg == "--text-template") {
            options.text_templates.push_back(require_value(argc, argv, i, "--text-template"));
        } else if (arg == "--encode-text-only") {
            options.encode_text_only = true;
        } else if (arg == "--token-output") {
            options.token_output = require_value(argc, argv, i, "--token-output");
        } else if (arg == "--flash-attention") {
            options.flash_attention = true;
        } else if (arg == "--no-flash-attention") {
            options.flash_attention = false;
        } else if (arg == "--no-compact-pairs") {
            options.compact_pairs = false;
        } else if (arg == "--matmul-sampler") {
            options.matmul_sampler = true;
        } else if (arg == "--matmul-geometry") {
            options.matmul_geometry = true;
        } else if (arg == "--no-fused-vocabulary") {
            options.fused_vocabulary = false;
        } else if (arg == "--no-cache-positions") {
            options.cache_positions = false;
        } else if (arg == "--quant-f32-input") {
            options.quant_f32_input = true;
        } else if (arg == "--strict-f32") {
            options.strict_f32 = true;
        } else if (arg == "--no-cuda-graphs") {
            options.runtime.cuda_disable_graphs = true;
        } else if (arg == "--no-retain-features") {
            options.retain_features = false;
        } else if (arg == "--raw-logits") {
            options.raw_logits = true;
        } else if (arg == "--preprocess-only") {
            options.preprocess_only = true;
        } else if (arg == "--backend") {
            options.backend = require_value(argc, argv, i, "--backend");
        } else if (arg == "--device-index") {
            options.device_index = parse_size(require_value(argc, argv, i, "--device-index"), "--device-index");
        } else if (arg == "--backend-profile") {
            options.runtime.vk_perf_logger = true;
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

void ModelOptions::validate_model() const {
    if (backend != "cpu" && backend != "cuda" && backend != "vulkan") {
        throw std::invalid_argument("--backend must be cpu, cuda or vulkan");
    }
    if (quant_f32_input && backend != "cuda")
        throw std::invalid_argument("--quant-f32-input requires --backend cuda");
    if (!max_boxes || max_boxes > 10000 || !topk) throw std::invalid_argument("invalid model object or output limits");
    if (threads > static_cast<std::size_t>(std::numeric_limits<int>::max())) throw std::invalid_argument("thread count exceeds integer range");
    if (backend == "cpu" && device_index != 0) throw std::invalid_argument("CPU device index must be zero");
}

void Options::validate() const {
    validate_model();
    if (repeats == 0) throw std::invalid_argument("--repeats must be greater than zero");
    if (max_boxes == 0) throw std::invalid_argument("--max-boxes must be greater than zero");
    if (!self_test && !preprocess_only && !encode_text_only && model.empty()) throw std::invalid_argument("--model is required");
    if (!texts.empty() && (text_student.empty() || tokenizer.empty()))
        throw std::invalid_argument("--texts requires --text-student and --tokenizer");
    if (!texts.empty() && !vocabulary.empty())
        throw std::invalid_argument("--texts and --vocabulary are mutually exclusive");
    if (encode_text_only && (texts.empty() || output.empty()))
        throw std::invalid_argument("--encode-text-only requires --texts and --output");
    if (preprocess_only && output.empty()) throw std::invalid_argument("--preprocess-only requires --output");
    if (!self_test && !encode_text_only && input.empty()) throw std::invalid_argument("--input is required");
}

void Options::print_help(const char * program) {
    std::printf("Usage: %s --model MODEL.gguf --input INPUT.raim|DIRECTORY [options]\n", program);
    std::printf("  --backend cpu|cuda|vulkan   execution backend (default cpu)\n");
    std::printf("  --no-cache-positions        recompute fixed scene positions (ablation)\n");
    std::printf("  --quant-f32-input           CUDA quantized weights with F32 activation GEMM\n");
    std::printf("  --strict-f32                preserve F32 matrix operands; use F32 attention\n");
    std::printf("  --no-cuda-graphs            disable CUDA graph capture/replay (ablation)\n");
    std::printf("  --no-retain-features        round-trip the patch map through host (ablation)\n");
    std::printf("  --output PATH               write calibrated predictions (JSON)\n");
    std::printf("  --raw-logits                write RAFO tensors for numerical comparison\n");
    std::printf("  --preprocess-only           convert image requests (.raip) to normalized RAIM\n");
    std::printf("  --dump-directory PATH       write intermediate F32 tensors\n");
    std::printf("  --vocabulary PATH           replace vocabulary with [V,512] F32 embeddings\n");
    std::printf("  --vocabulary-names PATH     JSON predicate names for replacement embeddings\n");
    std::printf("  --predicate-type-map PATH   JSON corpus spatial flags for decomposition\n");
    std::printf("  --decompose --topk N        independently ranked spatial/semantic graphs\n");
    std::printf("  --texts PATH               JSON free-text predicates; native student encoding\n");
    std::printf("  --text-student PATH --tokenizer PATH   student GGUF and tokenizer.json\n");
    std::printf("  --text-template STRING      ensemble template with {p}; repeatable\n");
    std::printf("  --encode-text-only          write [V,512] F32 embeddings without an image\n");
    std::printf("  --token-output PATH         write unpadded tokenizer IDs as JSON\n");
    std::printf("  --flash-attention           use backend flash attention\n");
    std::printf("  --no-flash-attention        use explicit F32 attention\n");
    std::printf("  --no-compact-pairs          retain padded computation for diagnostics\n");
    std::printf("  --matmul-sampler            use the historical dot-product diagnostic\n");
    std::printf("  --matmul-geometry           use Vulkan GEMM geometry scorer (ablation)\n");
    std::printf("  --no-fused-vocabulary       project vocabulary branches separately\n");
    std::printf("  --max-boxes N               maximum objects (default 60)\n");
    std::printf("  --device-index N            explicit backend device index (default 0)\n");
    std::printf("  --backend-profile           Vulkan kernel performance report\n");
    std::printf("  --threads N                 CPU worker count (0 = ggml default)\n");
    std::printf("  --warmup N --repeats N     benchmark controls\n");
    std::printf("  --profile                   print timing and tensor details\n");
    std::printf("  --self-test                 run a backend-independent graph test\n");
}

} // namespace relateanything
