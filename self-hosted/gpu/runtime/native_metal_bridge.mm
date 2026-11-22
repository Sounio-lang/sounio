// C++23/Objective-C++ bridge: all kernel arithmetic runs on Metal, never a CPU
// fallback. Eight-byte scalar/array storage preserves source binary64 bits.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>
#include "softfloat64_source.inc"

static constexpr char SF64_WRAPPER[] = R"(
struct sf64 { ulong bits; };
sf64 sf64_bits(ulong bits) { return sf64{bits}; }
sf64 sf64_from_i64(long value) { return sf64_bits(__softfloat64_cvt_i64_to_f64(value, 0u)); }
sf64 operator+(sf64 a, sf64 b) { return sf64_bits(__softfloat64_fadd(a.bits,b.bits,0u)); }
sf64 operator-(sf64 a, sf64 b) { return sf64_bits(__softfloat64_fsub(a.bits,b.bits,0u)); }
sf64 operator*(sf64 a, sf64 b) { return sf64_bits(__softfloat64_fmul(a.bits,b.bits,0u)); }
sf64 operator/(sf64 a, sf64 b) { return sf64_bits(__softfloat64_fdiv(a.bits,b.bits,0u)); }
sf64 operator-(sf64 a) { return sf64_bits(a.bits ^ 0x8000000000000000ul); }
bool operator==(sf64 a, sf64 b) { return __softfloat64_feq(a.bits,b.bits); }
bool operator!=(sf64 a, sf64 b) { return !__softfloat64_feq(a.bits,b.bits); }
bool operator<(sf64 a, sf64 b) { return __softfloat64_flt(a.bits,b.bits); }
bool operator<=(sf64 a, sf64 b) { return __softfloat64_fle(a.bits,b.bits); }
bool operator>(sf64 a, sf64 b) { return __softfloat64_fgt(a.bits,b.bits); }
bool operator>=(sf64 a, sf64 b) { return __softfloat64_fge(a.bits,b.bits); }
)";

struct SounioMetalArg { uint64_t value, kind; };
// kind: 0=i64 scalar, 1=f64 scalar, 2=readonly f64 array ref,
// 3=mutable f64 array ref, 4=readonly i64 array ref, 5=mutable i64 array ref.
static int fail(const char* kernel, const char* phase, NSError* error = nil) {
    std::fprintf(stderr, "SOUNIO_METAL_FAILURE kernel=%s phase=%s detail=%s\n",
                 kernel ? kernel : "<null>", phase,
                 error ? error.localizedDescription.UTF8String : "unavailable or invalid argument");
    return 1;
}

extern "C" int64_t sounio_metal_launch(const char* source, const char* kernel,
                                      const SounioMetalArg* args, uint64_t argc,
                                      uint64_t lanes) {
    @autoreleasepool {
        if (!source || !kernel || !args || argc == 0 || argc > 31 || lanes > 0x7fffffff)
            return fail(kernel, "abi");
        if (lanes == 0) return 0;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return fail(kernel, "device");
        NSString* prefix = [[NSString stringWithUTF8String:SOUNIO_SF64_SOURCE]
                           stringByAppendingString:[NSString stringWithUTF8String:SF64_WRAPPER]];
        NSString* body = [NSString stringWithUTF8String:source];
        if (!prefix || !body) return fail(kernel, "source-encoding");
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        options.languageVersion = MTLLanguageVersion3_0;
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[prefix stringByAppendingString:body]
                                                    options:options error:&error];
        if (!library) return fail(kernel, "compile", error);
        id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:kernel]];
        if (!function) return fail(kernel, "function");
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) return fail(kernel, "pipeline", error);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!queue || !command || !encoder) return fail(kernel, "command");
        [encoder setComputePipelineState:pipeline];
        struct Buffer {
            uint64_t* header;
            __strong id<MTLBuffer> gpu;
            size_t bytes;
            bool writable;
        };
        std::vector<Buffer> buffers;
        for (uint64_t i = 0; i < argc; ++i) {
            if (args[i].kind <= 1) {
                [encoder setBytes:&args[i].value length:8 atIndex:i];
                continue;
            }
            if (args[i].kind > 5 || !args[i].value) return fail(kernel, "argument-kind");
            auto* ref = reinterpret_cast<uint64_t* const*>(args[i].value);
            auto* header = *ref;
            if (!header) return fail(kernel, "array-reference");
            const uint64_t count = header[2];
            if (count > (1ull << 31) || count > header[1] / 8 || lanes > count)
                return fail(kernel, "array-bounds");
            const size_t bytes = static_cast<size_t>(count * 8);
            auto existing = std::find_if(buffers.begin(), buffers.end(),
                                        [=](const Buffer& b) { return b.header == header; });
            if (existing == buffers.end()) {
                id<MTLBuffer> gpu = [device newBufferWithBytes:header + 4 length:bytes
                                                      options:MTLResourceStorageModeShared];
                if (!gpu) return fail(kernel, "buffer");
                buffers.push_back({header, gpu, bytes, (args[i].kind & 1) != 0});
                existing = buffers.end() - 1;
            } else {
                existing->writable = existing->writable || (args[i].kind & 1) != 0;
            }
            [encoder setBuffer:existing->gpu offset:0 atIndex:i];
        }
        const NSUInteger width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 128);
        if (!width) return fail(kernel, "threadgroup-size");
        [encoder dispatchThreads:MTLSizeMake(lanes, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) return fail(kernel, "execute", command.error);
        for (const auto& buffer : buffers)
            if (buffer.writable) std::memcpy(buffer.header + 4, buffer.gpu.contents, buffer.bytes);
        if (std::getenv("SOUNIO_METAL_TRACE"))
            std::fprintf(stderr, "SOUNIO_METAL_EXECUTED device=%s kernel=%s lanes=%llu policy=software_binary64\n",
                         device.name.UTF8String, kernel, static_cast<unsigned long long>(lanes));
        return 0;
    }
}
