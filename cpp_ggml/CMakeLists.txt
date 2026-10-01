cmake_minimum_required(VERSION 3.21)
project(relateanything_ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(RA_GGML_CUDA "Build the ggml CUDA backend" OFF)
option(RA_GGML_VULKAN "Build the ggml Vulkan backend" OFF)
option(RA_GGML_BUILD_TESTS "Build the C++ runtime self-test" ON)

if(RA_GGML_CUDA AND RA_GGML_VULKAN)
    message(FATAL_ERROR "Select one accelerator backend per build directory")
endif()

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${RA_GGML_CUDA} CACHE BOOL "" FORCE)
set(GGML_VULKAN ${RA_GGML_VULKAN} CACHE BOOL "" FORCE)
# This integration has no cuDNN path. Keep optional CUDA extras disabled.
set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA_FA ON CACHE BOOL "" FORCE)

set(RA_GGML_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml")
if(NOT EXISTS "${RA_GGML_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Initialize the pinned dependency with: git submodule update --init --recursive")
endif()

execute_process(
    COMMAND git -C "${RA_GGML_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE RA_GGML_HEAD
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE RA_GGML_GIT_RESULT)
if(NOT RA_GGML_GIT_RESULT EQUAL 0 OR
   NOT RA_GGML_HEAD STREQUAL "8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b")
    message(FATAL_ERROR "cpp_ggml requires ggml v0.21.0 (8599e0ea), found ${RA_GGML_HEAD}")
endif()

file(GLOB RA_GGML_PATCHES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/patches/*.patch")
list(SORT RA_GGML_PATCHES)
foreach(RA_GGML_PATCH IN LISTS RA_GGML_PATCHES)
    execute_process(
        COMMAND git -C "${RA_GGML_SOURCE_DIR}" apply --reverse --check "${RA_GGML_PATCH}"
        RESULT_VARIABLE RA_PATCH_REVERSED
        OUTPUT_QUIET ERROR_QUIET)
    if(NOT RA_PATCH_REVERSED EQUAL 0)
        execute_process(
            COMMAND git -C "${RA_GGML_SOURCE_DIR}" apply --check "${RA_GGML_PATCH}"
            RESULT_VARIABLE RA_PATCH_CHECK
            OUTPUT_VARIABLE RA_PATCH_OUTPUT
            ERROR_VARIABLE RA_PATCH_ERROR)
        if(NOT RA_PATCH_CHECK EQUAL 0)
            message(FATAL_ERROR "Cannot apply ${RA_GGML_PATCH}: ${RA_PATCH_OUTPUT} ${RA_PATCH_ERROR}")
        endif()
        execute_process(
            COMMAND git -C "${RA_GGML_SOURCE_DIR}" apply "${RA_GGML_PATCH}"
            RESULT_VARIABLE RA_PATCH_RESULT)
        if(NOT RA_PATCH_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to apply ${RA_GGML_PATCH}")
        endif()
    endif()
endforeach()

add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)
find_package(JPEG REQUIRED)

add_library(relateanything_ggml STATIC
    src/backend.cpp
    src/gguf_model.cpp
    src/full_model.cpp
    src/image_io.cpp
    src/inference.cpp)
target_include_directories(relateanything_ggml
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${RA_GGML_SOURCE_DIR}/examples")
target_link_libraries(relateanything_ggml PUBLIC ggml PRIVATE JPEG::JPEG)
target_compile_options(relateanything_ggml PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic>)

add_executable(relateanything-ggml src/main.cpp)
target_link_libraries(relateanything-ggml PRIVATE relateanything_ggml)

if(RA_GGML_BUILD_TESTS)
    enable_testing()
    add_test(NAME relateanything_ggml_self_test
        COMMAND $<TARGET_FILE:relateanything-ggml> --backend cpu --self-test)
    if(RA_GGML_CUDA)
        add_test(NAME relateanything_ggml_cuda_self_test
            COMMAND $<TARGET_FILE:relateanything-ggml> --backend cuda --self-test)
    elseif(RA_GGML_VULKAN)
        add_test(NAME relateanything_ggml_vulkan_self_test
            COMMAND $<TARGET_FILE:relateanything-ggml> --backend vulkan --self-test)
    endif()
endif()
