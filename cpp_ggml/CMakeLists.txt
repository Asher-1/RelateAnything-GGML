cmake_minimum_required(VERSION 3.21)
project(relateanything_ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(RA_GGML_CUDA "Build the ggml CUDA backend" OFF)
option(RA_GGML_CUDA_GRAPHS "Compile CUDA graph capture/replay support" ON)
option(RA_GGML_VULKAN "Build the ggml Vulkan backend" OFF)
option(RA_GGML_BUILD_TESTS "Build the C++ runtime self-test" ON)

if(RA_GGML_CUDA AND RA_GGML_VULKAN)
    message(FATAL_ERROR "Select one accelerator backend per build directory")
endif()

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${RA_GGML_CUDA} CACHE BOOL "" FORCE)
set(GGML_CUDA_GRAPHS ${RA_GGML_CUDA_GRAPHS} CACHE BOOL "" FORCE)
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
# Reconstruct the patched tree in an isolated Git index, then compare its
# contents and untracked files with the actual checkout. A reverse-check alone
# can miss extra edits outside the hunks of an otherwise applied patch.
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef RA_GGML_AUDIT_SUFFIX)
set(RA_GGML_AUDIT_INDEX "${CMAKE_CURRENT_BINARY_DIR}/ggml-patch-audit-${RA_GGML_AUDIT_SUFFIX}.index")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "GIT_INDEX_FILE=${RA_GGML_AUDIT_INDEX}"
        git -C "${RA_GGML_SOURCE_DIR}" read-tree HEAD
    RESULT_VARIABLE RA_GGML_AUDIT_RESULT)
if(NOT RA_GGML_AUDIT_RESULT EQUAL 0)
    message(FATAL_ERROR "Cannot initialize the GGML patch audit index")
endif()
foreach(RA_GGML_PATCH IN LISTS RA_GGML_PATCHES)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env "GIT_INDEX_FILE=${RA_GGML_AUDIT_INDEX}"
            git -C "${RA_GGML_SOURCE_DIR}" apply --cached "${RA_GGML_PATCH}"
        RESULT_VARIABLE RA_GGML_AUDIT_RESULT)
    if(NOT RA_GGML_AUDIT_RESULT EQUAL 0)
        message(FATAL_ERROR "Cannot replay ${RA_GGML_PATCH} onto the clean GGML index")
    endif()
endforeach()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "GIT_INDEX_FILE=${RA_GGML_AUDIT_INDEX}"
        git -C "${RA_GGML_SOURCE_DIR}" diff --quiet --no-ext-diff
    RESULT_VARIABLE RA_GGML_TARGET_MATCH)
# Later patches can change an earlier hunk's context. The complete target tree
# is the idempotence check; individual forward/reverse checks handle partial trees.
if(NOT RA_GGML_TARGET_MATCH EQUAL 0)
    foreach(RA_GGML_PATCH IN LISTS RA_GGML_PATCHES)
        execute_process(
            COMMAND git -C "${RA_GGML_SOURCE_DIR}" apply --check "${RA_GGML_PATCH}"
            RESULT_VARIABLE RA_PATCH_FORWARD)
        if(RA_PATCH_FORWARD EQUAL 0)
            execute_process(
                COMMAND git -C "${RA_GGML_SOURCE_DIR}" apply "${RA_GGML_PATCH}"
                RESULT_VARIABLE RA_PATCH_RESULT ERROR_VARIABLE RA_PATCH_ERROR)
            if(NOT RA_PATCH_RESULT EQUAL 0)
                message(FATAL_ERROR "Cannot apply ${RA_GGML_PATCH}: ${RA_PATCH_ERROR}")
            endif()
            continue()
        endif()
        execute_process(
            COMMAND git -C "${RA_GGML_SOURCE_DIR}" apply --reverse --check "${RA_GGML_PATCH}"
            RESULT_VARIABLE RA_PATCH_REVERSED OUTPUT_QUIET ERROR_QUIET)
        if(NOT RA_PATCH_REVERSED EQUAL 0)
            message(FATAL_ERROR "GGML tree is neither pristine nor patched by ${RA_GGML_PATCH}")
        endif()
    endforeach()
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "GIT_INDEX_FILE=${RA_GGML_AUDIT_INDEX}"
        git -C "${RA_GGML_SOURCE_DIR}" diff --quiet
    RESULT_VARIABLE RA_GGML_AUDIT_RESULT)
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "GIT_INDEX_FILE=${RA_GGML_AUDIT_INDEX}"
        git -C "${RA_GGML_SOURCE_DIR}" ls-files --others
    OUTPUT_VARIABLE RA_GGML_EXTRA_FILES
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE RA_GGML_EXTRA_RESULT)
file(REMOVE "${RA_GGML_AUDIT_INDEX}")
if(NOT RA_GGML_AUDIT_RESULT EQUAL 0 OR
   NOT RA_GGML_EXTRA_RESULT EQUAL 0 OR
   NOT RA_GGML_EXTRA_FILES STREQUAL "")
    message(FATAL_ERROR
        "GGML checkout differs from the exact pinned-commit-plus-patches tree. "
        "Put every GGML source change in cpp_ggml/third_party/patches. "
        "Unexpected files: ${RA_GGML_EXTRA_FILES}")
endif()

add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)
find_package(JPEG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)
find_package(ICU COMPONENTS uc i18n REQUIRED)

add_library(relateanything_ggml STATIC
    src/c_api.cpp
    src/backend.cpp
    src/gguf_model.cpp
    src/full_model.cpp
    src/image_io.cpp
    src/inference.cpp)
target_include_directories(relateanything_ggml
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${RA_GGML_SOURCE_DIR}/examples")
target_link_libraries(relateanything_ggml PUBLIC ggml PRIVATE JPEG::JPEG jsoncpp_lib ICU::uc ICU::i18n)
target_include_directories(relateanything_ggml PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_compile_options(relateanything_ggml PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic>)

add_executable(relateanything-ggml src/main.cpp)
target_link_libraries(relateanything-ggml PRIVATE relateanything_ggml jsoncpp_lib)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
file(GLOB RA_PROVENANCE_INPUTS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/relateanything/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/relateanything/*.h")
set(RA_BACKEND_NAME cpu)
if(RA_GGML_CUDA)
    set(RA_BACKEND_NAME cuda)
elseif(RA_GGML_VULKAN)
    set(RA_BACKEND_NAME vulkan)
endif()
set(RA_BUILD_INFO "${CMAKE_CURRENT_BINARY_DIR}/generated/ra_build_info.hpp")
string(TOUPPER "${CMAKE_BUILD_TYPE}" RA_BUILD_TYPE_UPPER)
add_custom_command(OUTPUT "${RA_BUILD_INFO}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/provenance.py"
        --header "${RA_BUILD_INFO}" --backend "${RA_BACKEND_NAME}"
        --compiler "${CMAKE_CXX_COMPILER}" --build-type "${CMAKE_BUILD_TYPE}"
        --cache "${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt"
        "--flags=${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${RA_BUILD_TYPE_UPPER}}"
    DEPENDS ${RA_PROVENANCE_INPUTS} ${RA_GGML_PATCHES}
        "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/provenance.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/../relsgg/corpus_type_map.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/clip_tokenizer.hpp"
        "${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt"
    VERBATIM)
target_sources(relateanything-ggml PRIVATE "${RA_BUILD_INFO}")
target_include_directories(relateanything-ggml PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_sources(relateanything_ggml PRIVATE "${RA_BUILD_INFO}")
target_include_directories(relateanything_ggml PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

if(RA_GGML_BUILD_TESTS)
    enable_testing()
    add_executable(relateanything-capi-header tests/c_api_header.c)
    target_link_libraries(relateanything-capi-header PRIVATE relateanything_ggml)
    set_target_properties(relateanything-capi-header PROPERTIES LINKER_LANGUAGE CXX)
    add_test(NAME relateanything_capi_c_header COMMAND relateanything-capi-header)
    add_executable(relateanything-runtime-options tests/runtime_options_contract.cpp)
    target_link_libraries(relateanything-runtime-options PRIVATE ggml-base)
    add_test(NAME relateanything_runtime_options COMMAND relateanything-runtime-options)
    add_test(NAME relateanything_runtime_defaults_frozen COMMAND relateanything-runtime-options default)
    add_executable(relateanything-capi-contract tests/c_api_contract.cpp)
    target_link_libraries(relateanything-capi-contract PRIVATE relateanything_ggml JPEG::JPEG)
    add_test(NAME relateanything_capi_contract COMMAND relateanything-capi-contract)
    add_test(NAME relateanything_ggml_self_test
        COMMAND $<TARGET_FILE:relateanything-ggml> --backend cpu --self-test)
    if(RA_GGML_CUDA)
        add_executable(relateanything-softmax-probe tests/strict_softmax_probe.cpp)
        target_link_libraries(relateanything-softmax-probe PRIVATE ggml ggml-cuda)
        add_test(NAME relateanything_ggml_cuda_self_test
            COMMAND $<TARGET_FILE:relateanything-ggml> --backend cuda --self-test)
    elseif(RA_GGML_VULKAN)
        add_test(NAME relateanything_ggml_vulkan_self_test
            COMMAND $<TARGET_FILE:relateanything-ggml> --backend vulkan --self-test)
    endif()
endif()
