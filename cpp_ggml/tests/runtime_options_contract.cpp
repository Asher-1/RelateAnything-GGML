#include "ggml-runtime-options.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
int main(int argc,char **) {
    auto o=ggml_runtime_default_options();
    auto check=[](bool ok) { if(!ok) throw std::runtime_error("runtime options contract"); };
    try {
        check(!ggml_runtime_configure(nullptr));
        auto bad=o;bad.abi_version=0;check(!ggml_runtime_configure(&bad));
        bad=o;bad.vk_max_nodes_per_submit=0;check(!ggml_runtime_configure(&bad));
        bad=o;std::memset(bad.backend_path,'x',sizeof(bad.backend_path));check(!ggml_runtime_configure(&bad));
        if(argc>1) {
            // Backend discovery before configuration freezes the default policy.
            check(ggml_runtime_get_options()->cpu_disable_fusion==false);
            o.cpu_disable_fusion=true;check(!ggml_runtime_configure(&o));
            o=ggml_runtime_default_options();check(ggml_runtime_configure(&o));
        } else {
            o.cpu_disable_fusion=true;check(ggml_runtime_configure(&o));
            check(ggml_runtime_get_options()->cpu_disable_fusion);
            check(ggml_runtime_configure(&o));
            o.cpu_disable_fusion=false;check(!ggml_runtime_configure(&o));
            check(ggml_runtime_get_options()->cpu_disable_fusion);
        }
        std::cout<<"RUNTIME_OPTIONS_CONTRACT PASS\n";
    } catch(const std::exception & e) { std::cerr<<e.what()<<'\n';return 1; }
}
