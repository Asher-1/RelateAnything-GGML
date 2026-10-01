#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace relateanything {

struct Options {
    std::filesystem::path model;
    std::filesystem::path input;
    std::filesystem::path output;
    std::filesystem::path dump_directory;
    std::filesystem::path vocabulary;
    std::string backend = "cpu";
    std::size_t threads = 0;
    std::size_t warmup = 2;
    std::size_t repeats = 10;
    std::size_t max_boxes = 60;
    bool flash_attention = true;
    bool raw_logits = false;
    bool preprocess_only = false;
    bool profile = false;
    bool self_test = false;

    static Options from_args(int argc, char ** argv);
    static void print_help(const char * program);
    void validate() const;
};

} // namespace relateanything
