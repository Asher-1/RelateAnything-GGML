#include "relateanything/c_api.h"
#include <stdio.h>
int main(void) {
    ra_options options=ra_default_options();
    ra_runtime_options runtime=ra_default_runtime_options();
    ra_request request=ra_default_request();
    if(options.struct_size!=sizeof(options) || runtime.struct_size!=sizeof(runtime) ||
       request.struct_size!=sizeof(request) || options.abi_version!=RA_ABI_VERSION ||
       runtime.abi_version!=RA_ABI_VERSION || runtime.op_offload_min_batch!=32 ||
       !options.strict_f32 || runtime.cuda_disable_graphs || runtime.vk_disable_f16) return 1;
    puts("C_API_HEADER PASS");
    return 0;
}
