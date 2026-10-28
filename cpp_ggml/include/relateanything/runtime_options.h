#pragma once
#include <stdbool.h>
#include <stdint.h>
#ifdef __cplusplus
#define RA_DEFAULT(value) = value
#else
#define RA_DEFAULT(value)
#endif
/* Process runtime options, frozen before backend discovery. */
typedef struct ra_runtime_options {
    uint32_t struct_size RA_DEFAULT(sizeof(ra_runtime_options));
    uint32_t abi_version RA_DEFAULT(1);
    bool no_backtrace RA_DEFAULT(false);
    bool backtrace_lldb RA_DEFAULT(false);
    bool cpu_disable_fusion RA_DEFAULT(false);
    bool cuda_enable_unified_memory RA_DEFAULT(false);
    bool cuda_p2p RA_DEFAULT(false);
    bool cuda_no_pinned RA_DEFAULT(false);
    bool cuda_disable_fusion RA_DEFAULT(false);
    bool cuda_graph_opt RA_DEFAULT(false);
    bool cuda_register_host RA_DEFAULT(false);
    bool cuda_disable_graphs RA_DEFAULT(false);
    bool vk_prefer_host_memory RA_DEFAULT(false);
    bool vk_disable_host_visible_vidmem RA_DEFAULT(false);
    bool vk_allow_sysmem_fallback RA_DEFAULT(false);
    bool vk_disable_graph_optimize RA_DEFAULT(false);
    bool vk_disable_coopmat RA_DEFAULT(false);
    bool vk_disable_coopmat2 RA_DEFAULT(false);
    bool vk_disable_coopmat2_decode_vector RA_DEFAULT(false);
    bool vk_disable_integer_dot_product RA_DEFAULT(false);
    bool vk_disable_bfloat16 RA_DEFAULT(false);
    bool vk_disable_dot2 RA_DEFAULT(false);
    bool vk_enable_memory_priority RA_DEFAULT(false);
    bool vk_disable_async RA_DEFAULT(false);
    bool vk_disable_f16 RA_DEFAULT(false);
    bool vk_allow_graphics_queue RA_DEFAULT(false);
    bool vk_disable_ocp_fp4 RA_DEFAULT(false);
    bool vk_disable_multi_add RA_DEFAULT(false);
    bool vk_async_use_transfer_queue RA_DEFAULT(false);
    bool vk_serialize_submissions RA_DEFAULT(false);
    bool vk_disable_fusion RA_DEFAULT(false);
    bool vk_disable_mmvq RA_DEFAULT(false);
    bool vk_force_mmvq RA_DEFAULT(false);
    bool vk_debug_markers RA_DEFAULT(false);
    bool vk_perf_logger RA_DEFAULT(false);
    bool vk_perf_logger_concurrent RA_DEFAULT(false);
    bool vk_sync_logger RA_DEFAULT(false);
    bool vk_memory_logger RA_DEFAULT(false);
    int32_t sched_debug RA_DEFAULT(0);
    int32_t sched_debug_realloc RA_DEFAULT(-1);
    int32_t meta_debug RA_DEFAULT(0);
    int32_t cuda_devices RA_DEFAULT(0);
    int32_t cuda_pdl RA_DEFAULT(-1);
    int32_t op_offload_min_batch RA_DEFAULT(32);
    int32_t vk_max_nodes_per_submit RA_DEFAULT(100);
    int32_t vk_perf_logger_frequency RA_DEFAULT(1);
    int32_t vulkan_skip_checks RA_DEFAULT(0);
    int32_t vulkan_output_tensor RA_DEFAULT(0);
    uint64_t vk_force_max_allocation_size RA_DEFAULT(0);
    uint64_t vk_force_max_buffer_size RA_DEFAULT(0);
    uint64_t vk_suballocation_block_size RA_DEFAULT(0);
    uint64_t cuda_ar_copy_threshold RA_DEFAULT(1048576);
    uint64_t cuda_ar_copy_chunk_bytes RA_DEFAULT(0);
    uint64_t cuda_ar_bf16_threshold RA_DEFAULT(1);
    char backend_path[1024] RA_DEFAULT({});
    char cuda_allreduce[16] RA_DEFAULT({});
    char cuda_cublas_compute_type[16] RA_DEFAULT({});
    char vk_visible_devices[256] RA_DEFAULT({});
    char vk_pipeline_stats[256] RA_DEFAULT({});
} ra_runtime_options;
#undef RA_DEFAULT
