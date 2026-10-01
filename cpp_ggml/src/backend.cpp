#include "relateanything/backend.hpp"

#include "ggml-cpu.h"

#include <stdexcept>

namespace relateanything {

Backend::Backend(const Options & options) : name(options.backend) {
    ggml_backend_load_all();
    cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        throw std::runtime_error("ggml CPU backend is unavailable");
    }
    if (options.threads > 0) {
        ggml_backend_cpu_set_n_threads(cpu, static_cast<int>(options.threads));
    }
    if (name == "cpu") {
        accelerator = cpu;
    } else if (name == "cuda") {
        accelerator = ggml_backend_init_by_name("CUDA0", nullptr);
    } else if (name == "vulkan") {
        accelerator = ggml_backend_init_by_name("Vulkan0", nullptr);
    } else {
        throw std::invalid_argument("backend must be cpu, cuda or vulkan: " + name);
    }
    if (!accelerator) {
        throw std::runtime_error("requested ggml backend is unavailable: " + name);
    }
    ggml_backend_t backends[2] = {accelerator, cpu};
    const std::size_t count = accelerator == cpu ? 1 : 2;
    // The full RelateAnything graph contains the backbone, two relation
    // transformer stacks and the vocabulary projection.  Keep the scheduler
    // capacity above the largest graph so CUDA/Vulkan do not silently fall
    // back to a truncated allocation.
    scheduler = ggml_backend_sched_new(backends, nullptr, count,
                                       8192, false, true);
    if (!scheduler) {
        throw std::runtime_error("cannot create ggml scheduler");
    }
}

Backend::~Backend() {
    if (scheduler) ggml_backend_sched_free(scheduler);
    if (accelerator && accelerator != cpu) ggml_backend_free(accelerator);
    if (cpu) ggml_backend_free(cpu);
}

} // namespace relateanything
