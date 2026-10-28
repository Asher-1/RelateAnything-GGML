#include "relateanything/backend.hpp"

#include "ggml-cpu.h"
#include "ggml-runtime-options.h"
#include <cstring>

#include <stdexcept>

namespace relateanything {

Backend::Backend(const ModelOptions & options) : name(options.backend) {
    options.validate_model();
    binding=options.backend_binding;
    if (binding.struct_size) {
        if (binding.struct_size!=sizeof(binding) || binding.abi_version!=1 ||
            !binding.cpu_handle || !binding.accelerator_handle ||
            !binding.retain_lease || !binding.release_lease)
            throw std::invalid_argument("invalid host backend binding");
        cpu=static_cast<ggml_backend_t>(binding.cpu_handle);
        accelerator=static_cast<ggml_backend_t>(binding.accelerator_handle);
        ggml_backend_t handles[2]={accelerator,cpu};
        scheduler=ggml_backend_sched_new(handles,nullptr,accelerator==cpu ? 1 : 2,8192,false,true);
        if (!scheduler) throw std::runtime_error("cannot create host-bound ggml scheduler");
        binding.retain_lease(binding.lease_user);lease_retained=true;
        return;
    }
    if (options.runtime.struct_size != sizeof(options.runtime) || options.runtime.abi_version != 1)
        throw std::invalid_argument("invalid runtime options ABI");
    auto runtime = ggml_runtime_default_options();
    runtime.no_backtrace = options.runtime.no_backtrace;
    runtime.backtrace_lldb = options.runtime.backtrace_lldb;
    runtime.cpu_disable_fusion = options.runtime.cpu_disable_fusion;
    runtime.cuda_enable_unified_memory = options.runtime.cuda_enable_unified_memory;
    runtime.cuda_p2p = options.runtime.cuda_p2p;
    runtime.cuda_no_pinned = options.runtime.cuda_no_pinned;
    runtime.cuda_disable_fusion = options.runtime.cuda_disable_fusion;
    runtime.cuda_graph_opt = options.runtime.cuda_graph_opt;
    runtime.cuda_register_host = options.runtime.cuda_register_host;
    runtime.cuda_disable_graphs = options.runtime.cuda_disable_graphs;
    runtime.vk_prefer_host_memory = options.runtime.vk_prefer_host_memory;
    runtime.vk_disable_host_visible_vidmem = options.runtime.vk_disable_host_visible_vidmem;
    runtime.vk_allow_sysmem_fallback = options.runtime.vk_allow_sysmem_fallback;
    runtime.vk_disable_graph_optimize = options.runtime.vk_disable_graph_optimize;
    runtime.vk_disable_coopmat = options.runtime.vk_disable_coopmat;
    runtime.vk_disable_coopmat2 = options.runtime.vk_disable_coopmat2;
    runtime.vk_disable_coopmat2_decode_vector = options.runtime.vk_disable_coopmat2_decode_vector;
    runtime.vk_disable_integer_dot_product = options.runtime.vk_disable_integer_dot_product;
    runtime.vk_disable_bfloat16 = options.runtime.vk_disable_bfloat16;
    runtime.vk_disable_dot2 = options.runtime.vk_disable_dot2;
    runtime.vk_enable_memory_priority = options.runtime.vk_enable_memory_priority;
    runtime.vk_disable_async = options.runtime.vk_disable_async;
    runtime.vk_disable_f16 = options.runtime.vk_disable_f16;
    runtime.vk_allow_graphics_queue = options.runtime.vk_allow_graphics_queue;
    runtime.vk_disable_ocp_fp4 = options.runtime.vk_disable_ocp_fp4;
    runtime.vk_disable_multi_add = options.runtime.vk_disable_multi_add;
    runtime.vk_async_use_transfer_queue = options.runtime.vk_async_use_transfer_queue;
    runtime.vk_serialize_submissions = options.runtime.vk_serialize_submissions;
    runtime.vk_disable_fusion = options.runtime.vk_disable_fusion;
    runtime.vk_disable_mmvq = options.runtime.vk_disable_mmvq;
    runtime.vk_force_mmvq = options.runtime.vk_force_mmvq;
    runtime.vk_debug_markers = options.runtime.vk_debug_markers;
    runtime.vk_perf_logger = options.runtime.vk_perf_logger;
    runtime.vk_perf_logger_concurrent = options.runtime.vk_perf_logger_concurrent;
    runtime.vk_sync_logger = options.runtime.vk_sync_logger;
    runtime.vk_memory_logger = options.runtime.vk_memory_logger;
    runtime.sched_debug = options.runtime.sched_debug;
    runtime.sched_debug_realloc = options.runtime.sched_debug_realloc;
    runtime.meta_debug = options.runtime.meta_debug;
    runtime.cuda_devices = options.runtime.cuda_devices;
    runtime.cuda_pdl = options.runtime.cuda_pdl;
    runtime.op_offload_min_batch = options.runtime.op_offload_min_batch;
    runtime.vk_max_nodes_per_submit = options.runtime.vk_max_nodes_per_submit;
    runtime.vk_perf_logger_frequency = options.runtime.vk_perf_logger_frequency;
    runtime.vulkan_skip_checks = options.runtime.vulkan_skip_checks;
    runtime.vulkan_output_tensor = options.runtime.vulkan_output_tensor;
    runtime.vk_force_max_allocation_size = options.runtime.vk_force_max_allocation_size;
    runtime.vk_force_max_buffer_size = options.runtime.vk_force_max_buffer_size;
    runtime.vk_suballocation_block_size = options.runtime.vk_suballocation_block_size;
    std::memcpy(runtime.backend_path, options.runtime.backend_path, sizeof(runtime.backend_path));
    std::memcpy(runtime.cuda_allreduce, options.runtime.cuda_allreduce, sizeof(runtime.cuda_allreduce));
    std::memcpy(runtime.cuda_cublas_compute_type, options.runtime.cuda_cublas_compute_type, sizeof(runtime.cuda_cublas_compute_type));
    std::memcpy(runtime.vk_visible_devices, options.runtime.vk_visible_devices, sizeof(runtime.vk_visible_devices));
    std::memcpy(runtime.vk_pipeline_stats, options.runtime.vk_pipeline_stats, sizeof(runtime.vk_pipeline_stats));
    runtime.cuda_ar_copy_threshold = options.runtime.cuda_ar_copy_threshold;
    runtime.cuda_ar_copy_chunk_bytes = options.runtime.cuda_ar_copy_chunk_bytes;
    runtime.cuda_ar_bf16_threshold = options.runtime.cuda_ar_bf16_threshold;
    if (!ggml_runtime_configure(&runtime)) throw std::invalid_argument("invalid or conflicting GGML runtime options; configure before backend discovery");
    ggml_backend_load_all();
    try {
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
        accelerator = ggml_backend_init_by_name(("CUDA" + std::to_string(options.device_index)).c_str(), nullptr);
    } else if (name == "vulkan") {
        accelerator = ggml_backend_init_by_name(("Vulkan" + std::to_string(options.device_index)).c_str(), nullptr);
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
    } catch (...) {
        if (scheduler) ggml_backend_sched_free(scheduler);
        if (accelerator && accelerator != cpu) ggml_backend_free(accelerator);
        if (cpu) ggml_backend_free(cpu);
        throw;
    }
}

Backend::~Backend() {
    if (scheduler) ggml_backend_sched_free(scheduler);
    if (lease_retained) { binding.release_lease(binding.lease_user); return; }
    if (accelerator && accelerator != cpu) ggml_backend_free(accelerator);
    if (cpu) ggml_backend_free(cpu);
}

} // namespace relateanything
