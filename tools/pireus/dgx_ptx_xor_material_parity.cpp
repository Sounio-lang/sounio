// C++ is MATERIAL_PARITY only. Expected values come from frozen Sounio output.
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using CUdevice=int; using CUcontext=void*; using CUmodule=void*; using CUfunction=void*;
using CUdeviceptr=std::uint64_t; using CUresult=int;
using cuInit_t=CUresult(*)(unsigned); using cuDeviceGet_t=CUresult(*)(CUdevice*,int);
using cuDeviceGetName_t=CUresult(*)(char*,int,CUdevice); using cuDeviceComputeCapability_t=CUresult(*)(int*,int*,CUdevice);
using cuCtxCreate_t=CUresult(*)(CUcontext*,unsigned,CUdevice); using cuCtxDestroy_t=CUresult(*)(CUcontext);
using cuModuleLoadData_t=CUresult(*)(CUmodule*,const void*); using cuModuleGetFunction_t=CUresult(*)(CUfunction*,CUmodule,const char*);
using cuModuleUnload_t=CUresult(*)(CUmodule); using cuMemAlloc_t=CUresult(*)(CUdeviceptr*,std::size_t);
using cuMemFree_t=CUresult(*)(CUdeviceptr); using cuMemcpyHtoD_t=CUresult(*)(CUdeviceptr,const void*,std::size_t);
using cuMemcpyDtoH_t=CUresult(*)(void*,CUdeviceptr,std::size_t);
using cuLaunchKernel_t=CUresult(*)(CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void*,void**,void**);
using cuCtxSynchronize_t=CUresult(*)();

[[noreturn]] static void fail(const char* stage,CUresult rc=-1){
    std::fprintf(stderr,"result=FAIL stage=%s cuda_result=%d\n",stage,rc); std::exit(1);
}
template<class T> static T symbol(void* lib,const char* primary,const char* fallback=nullptr){
    void* value=dlsym(lib,primary); if(!value&&fallback)value=dlsym(lib,fallback); if(!value)fail(primary); return reinterpret_cast<T>(value);
}
static std::string read_all(const char* path){
    std::ifstream in(path,std::ios::binary); if(!in)fail("ptx_open"); return {std::istreambuf_iterator<char>(in),{}};
}
static std::vector<double> read_expected(const char* path){
    std::ifstream in(path); if(!in)fail("frozen_semantics_open"); std::vector<double> values; double value=0;
    while(in>>value) values.push_back(value);
    if(values.size()!=16) fail("frozen_semantics_shape");
    return values;
}

int main(int argc,char** argv){
    if(argc!=5){std::fprintf(stderr,"usage: %s PTX FROZEN SOURCE_SHA SEMANTICS_SHA\n",argv[0]);return 64;}
    const auto ptx=read_all(argv[1]); const auto expected=read_expected(argv[2]);
    double a[16],b[16],out[16]{}; for(int i=0;i<16;++i){a[i]=i+1;b[i]=17-i;}
    void* lib=dlopen("libcuda.so.1",RTLD_NOW); if(!lib)fail("dlopen_libcuda");
    auto init=symbol<cuInit_t>(lib,"cuInit"); auto device_get=symbol<cuDeviceGet_t>(lib,"cuDeviceGet");
    auto device_name=symbol<cuDeviceGetName_t>(lib,"cuDeviceGetName"); auto device_cc=symbol<cuDeviceComputeCapability_t>(lib,"cuDeviceComputeCapability");
    auto ctx_create=symbol<cuCtxCreate_t>(lib,"cuCtxCreate_v2","cuCtxCreate"); auto ctx_destroy=symbol<cuCtxDestroy_t>(lib,"cuCtxDestroy_v2","cuCtxDestroy");
    auto module_load=symbol<cuModuleLoadData_t>(lib,"cuModuleLoadData"); auto function_get=symbol<cuModuleGetFunction_t>(lib,"cuModuleGetFunction");
    auto module_unload=symbol<cuModuleUnload_t>(lib,"cuModuleUnload"); auto alloc=symbol<cuMemAlloc_t>(lib,"cuMemAlloc_v2","cuMemAlloc");
    auto free_device=symbol<cuMemFree_t>(lib,"cuMemFree_v2","cuMemFree"); auto htod=symbol<cuMemcpyHtoD_t>(lib,"cuMemcpyHtoD_v2","cuMemcpyHtoD");
    auto dtoh=symbol<cuMemcpyDtoH_t>(lib,"cuMemcpyDtoH_v2","cuMemcpyDtoH"); auto launch=symbol<cuLaunchKernel_t>(lib,"cuLaunchKernel");
    auto sync=symbol<cuCtxSynchronize_t>(lib,"cuCtxSynchronize");
    CUdevice device=0; CUcontext context=nullptr; CUmodule module=nullptr; CUfunction function=nullptr;
    CUdeviceptr da=0,db=0,dout=0; CUresult rc=0;
#define CUDA_OK(call,stage) do{rc=(call);if(rc!=0)fail(stage,rc);}while(0)
    CUDA_OK(init(0),"cuInit"); CUDA_OK(device_get(&device,0),"cuDeviceGet"); CUDA_OK(ctx_create(&context,0,device),"cuCtxCreate");
    CUDA_OK(module_load(&module,ptx.c_str()),"cuModuleLoadData"); CUDA_OK(function_get(&function,module,"sedenion_xor_product"),"cuModuleGetFunction");
    CUDA_OK(alloc(&da,sizeof(a)),"cuMemAlloc_a"); CUDA_OK(alloc(&db,sizeof(b)),"cuMemAlloc_b"); CUDA_OK(alloc(&dout,sizeof(out)),"cuMemAlloc_output");
    CUDA_OK(htod(da,a,sizeof(a)),"cuMemcpyHtoD_a"); CUDA_OK(htod(db,b,sizeof(b)),"cuMemcpyHtoD_b");
    void* params[]={&da,&db,&dout}; CUDA_OK(launch(function,1,1,1,16,1,1,0,nullptr,params,nullptr),"cuLaunchKernel");
    CUDA_OK(sync(),"cuCtxSynchronize"); CUDA_OK(dtoh(out,dout,sizeof(out)),"cuMemcpyDtoH");
    for(int i=0;i<16;++i)if(out[i]!=expected[static_cast<std::size_t>(i)]){
        std::fprintf(stderr,"result=FAIL lane=%d expected=%.17g actual=%.17g\n",i,expected[i],out[i]);return 1;
    }
    char name[128]{};int major=-1,minor=-1;(void)device_name(name,sizeof(name),device);(void)device_cc(&major,&minor,device);
    std::printf("result=PASS lanes=16 device=%s cc=%d.%d sounio_source_sha256=%s frozen_semantics_sha256=%s\n",name,major,minor,argv[3],argv[4]);
    (void)free_device(dout);(void)free_device(db);(void)free_device(da);(void)module_unload(module);(void)ctx_destroy(context);dlclose(lib);return 0;
}
