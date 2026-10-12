#include "relateanything/c_api.h"
#include "relateanything/full_model.hpp"
#include "relateanything/backend.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <jpeglib.h>
namespace {
void check(bool ok,const char * message) { if(!ok) throw std::runtime_error(message); }
int cancel_now(void *) { return 1; }
int cancel_after(void * count) { return ++*static_cast<int *>(count)>1; }
void retain(void * count) { ++*static_cast<int *>(count); }
void release(void * count) { --*static_cast<int *>(count); }
}
int main(int argc,char ** argv) {
    try {
        auto options=ra_default_options(); options.abi_version=0;
        auto * bad=ra_create(&options); check(bad && !ra_ready(bad) && std::strlen(ra_last_error(bad)),"bad ABI creation");
        ra_result * output=reinterpret_cast<ra_result *>(1);
        check(ra_infer_image(bad,nullptr,nullptr,&output)==-1 && !output,"failure must clear output"); ra_free(bad);
        options=ra_default_options(); options.strict_f32=RA_STRICT_F32_AUTO-1; options.model_path="/absent/relateanything-model.gguf";
        bad=ra_create(&options); check(bad && !ra_ready(bad) && std::strstr(ra_last_error(bad),"strict_f32"),"invalid strict mode"); ra_free(bad);
        options=ra_default_options(); options.model_path="/absent/relateanything-model.gguf";
        bad=ra_create(&options); check(bad && !ra_ready(bad),"failed load readiness"); ra_free(bad);
        // Test every format with padded row strides against identical RGB pixels.
        constexpr size_t width=37,height=29;
        std::vector<uint8_t> rgb(width*height*3);
        relateanything::ImageOptions regions; regions.boxes_xyxy={0,0,20,20,10,5,30,25};
        for(int format=1;format<=5;++format) {
            for(size_t i=0;i<width*height;++i) for(int c=0;c<3;++c)
                rgb[3*i+c]=static_cast<uint8_t>((i*7+(format==3 ? 0 : c*37))%256);
            auto reference=relateanything::prepare_image({rgb.data(),width,height,width*3,rgb.size(),relateanything::ImageFormat::rgb8},regions);
            const int channels=format==3 ? 1 : format==2 || format==5 ? 4 : 3;
            const size_t stride=width*channels+13;
            std::vector<uint8_t> padded(height*stride,217);
            for(size_t y=0;y<height;++y) for(size_t x=0;x<width;++x) for(int c=0;c<channels;++c)
                padded[y*stride+x*channels+c]=c==3 ? 255 : rgb[(y*width+x)*3+((format==4 || format==5) ? 2-c : c)];
            auto prepared=relateanything::prepare_image({padded.data(),width,height,stride,padded.size(),static_cast<relateanything::ImageFormat>(format)},regions);
            check(prepared.rgb_chw==reference.rgb_chw && prepared.boxes_cxcywh==reference.boxes_cxcywh,"format/stride preprocessing mismatch");
        }
        bool rejected=false;
        try { relateanything::prepare_image({rgb.data(),width,height,width*3,rgb.size()-1,relateanything::ImageFormat::rgb8},regions); }
        catch(const std::invalid_argument &) { rejected=true; }
        check(rejected,"truncated borrowed buffer accepted");
        if(argc==4) {
            // Real checkpoint, real JPEG, independent file and memory API paths.
            std::ifstream request_file(argv[3]); std::string magic,path;size_t count;
            request_file>>magic>>std::quoted(path)>>count;check(magic=="RAIP1","expected RAIP1 request");
            regions.boxes_xyxy.resize(count*4);for(auto & v:regions.boxes_xyxy) { double d;request_file>>d;v=static_cast<float>(d); }
            check(bool(request_file),"request read");
            FILE * file=std::fopen(path.c_str(),"rb");check(file,"JPEG open");
            jpeg_decompress_struct info{};jpeg_error_mgr error{};info.err=jpeg_std_error(&error);
            jpeg_create_decompress(&info);jpeg_stdio_src(&info,file);jpeg_read_header(&info,TRUE);info.out_color_space=JCS_RGB;jpeg_start_decompress(&info);
            std::vector<uint8_t> pixels(info.output_width*info.output_height*3);
            const int w=info.output_width,h=info.output_height;
            while(info.output_scanline<info.output_height) { JSAMPROW row=pixels.data()+info.output_scanline*w*3;jpeg_read_scanlines(&info,&row,1); }
            jpeg_finish_decompress(&info);jpeg_destroy_decompress(&info);std::fclose(file);
            ra_image_view image{pixels.data(),w,h,static_cast<size_t>(w)*3,pixels.size(),RA_RGB8};
            auto req=ra_default_request();req.object_count=count;req.boxes_xyxy=regions.boxes_xyxy.data();
            options=ra_default_options();options.model_path=argv[1];options.backend=argv[2];options.raw_logits=1;options.threads=4;
            auto * ctx=ra_create(&options);check(ra_ready(ctx),ra_last_error(ctx));
            // Both defaults resolve strict_f32 automatically from the loaded GGUF.
            relateanything::Options direct;direct.model=argv[1];direct.backend=argv[2];direct.raw_logits=true;direct.threads=4;
            relateanything::FullModel model(direct);const auto expected=model.infer(relateanything::read_image_request(argv[3]));
            check(ra_infer_image(ctx,&image,&req,&output)==0,ra_last_error(ctx));
            check(output->pair_count==expected.pair_count && output->predicate_count==expected.predicate_count,"typed dimensions");
            check(std::memcmp(output->predicate_logits,expected.predicate_logits.data(),expected.predicate_logits.size()*4)==0,"C/C++ raw logits differ");
            check(output->relation_count==expected.relations.size(),"typed relation count");
            check(output->timings.valid_fields==(RA_TIMING_PREPROCESS|RA_TIMING_E2E) && output->timings.e2e_ms>=output->timings.preprocess_ms,"truthful timing");
            ra_result_free(output);
            // A non-negative strict_f32 overrides the automatic rule.
            auto forced=options;forced.strict_f32=1;auto * fixed=ra_create(&forced);check(ra_ready(fixed),ra_last_error(fixed));
            relateanything::Options strict=direct;strict.strict_f32=true;strict.strict_f32_auto=false;
            relateanything::FullModel strict_model(strict);check(strict_model.strict_f32(),"forced strict_f32 was replaced");
            const auto expected_strict=strict_model.infer(relateanything::read_image_request(argv[3]));
            check(ra_infer_image(fixed,&image,&req,&output)==0,ra_last_error(fixed));
            check(std::memcmp(output->predicate_logits,expected_strict.predicate_logits.data(),expected_strict.predicate_logits.size()*4)==0,"forced strict raw logits differ");
            ra_result_free(output);ra_free(fixed);
            // The embedding host owns handles and lease-group serialization.
            relateanything::Backend host_backend(direct);int leases=0;
            ra_backend_binding binding{sizeof(ra_backend_binding),RA_ABI_VERSION,
                host_backend.accelerator,host_backend.cpu,&leases,retain,release};
            options.backend_binding=&binding;
            auto * borrowed=ra_create(&options);check(ra_ready(borrowed),ra_last_error(borrowed));
            check(leases==1,"host lease was not retained");
            check(ra_infer_image(borrowed,&image,&req,&output)==0,ra_last_error(borrowed));
            check(std::memcmp(output->predicate_logits,expected.predicate_logits.data(),expected.predicate_logits.size()*4)==0,"host binding raw logits differ");
            ra_result_free(output);ra_free(borrowed);check(leases==0,"host lease was not released");
            check(ggml_backend_get_device(host_backend.accelerator)!=nullptr,"borrowed backend was freed");
            req.cancelled=cancel_now;check(ra_infer_image(ctx,&image,&req,&output)==1 && !output,"cancel before");
            int calls=0;req.cancelled=cancel_after;req.user=&calls;check(ra_infer_image(ctx,&image,&req,&output)==1 && !output,"cancel after");
            req.cancelled=nullptr;image.data_bytes=1;check(ra_infer_image(ctx,&image,&req,&output)==-1 && !output,"invalid image capacity");
            image.data_bytes=pixels.size();check(ra_infer_image(ctx,&image,&req,&output)==0,ra_last_error(ctx));
            ra_free(ctx);check(output->relation_count && output->predicate_names[0] && output->predicate_logits[0]==expected.predicate_logits[0],"result lifetime after free");
            ra_result_free(output);
        }
        std::cout<<"C_API_CONTRACT PASS"<<std::endl;
    } catch(const std::exception & e) {std::cerr<<e.what()<<std::endl;return 1;}
}
