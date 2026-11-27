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
#include <unordered_map>
#include <vector>
#include <unistd.h>
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
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        } else {
            options.fastMathEnabled = NO;
        }
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

namespace {

struct MetalSessionState {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
};

struct MetalBufferState {
    int64_t session_id;
    id<MTLBuffer> buffer;
    void* raw_ptr;
};

struct MetalPipelineState {
    int64_t session_id;
    id<MTLComputePipelineState> pipeline;
};

struct MetalCommandState {
    int64_t session_id;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
};

struct MetalBridgeStore {
    std::mutex mu;
    int64_t next_id = 1;
    std::unordered_map<int64_t, MetalSessionState> sessions;
    std::unordered_map<int64_t, MetalBufferState> buffers;
    std::unordered_map<int64_t, MetalPipelineState> pipelines;
    std::unordered_map<int64_t, MetalCommandState> commands;
};

MetalBridgeStore& bridge_store() {
    static MetalBridgeStore* s = new MetalBridgeStore();
    return *s;
}

} // namespace

extern "C" int64_t sounio_metal_session_create() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fail("<session>", "device");
            return 0;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            fail("<session>", "queue");
            return 0;
        }
        auto& st = bridge_store();
        std::lock_guard<std::mutex> lock(st.mu);
        int64_t sid = st.next_id++;
        st.sessions.emplace(sid, MetalSessionState{device, queue});
        return sid;
    }
}

extern "C" int64_t sounio_metal_session_destroy(int64_t sess_id) {
    @autoreleasepool {
        auto& st = bridge_store();
        std::lock_guard<std::mutex> lock(st.mu);
        for (auto it = st.commands.begin(); it != st.commands.end();) {
            if (it->second.session_id == sess_id) {
                if (it->second.encoder) {
                    [it->second.encoder endEncoding];
                }
                it = st.commands.erase(it);
            }
            else ++it;
        }
        for (auto it = st.pipelines.begin(); it != st.pipelines.end();) {
            if (it->second.session_id == sess_id) it = st.pipelines.erase(it);
            else ++it;
        }
        for (auto it = st.buffers.begin(); it != st.buffers.end();) {
            if (it->second.session_id == sess_id) it = st.buffers.erase(it);
            else ++it;
        }
        st.sessions.erase(sess_id);
        return 0;
    }
}

extern "C" int64_t sounio_metal_buffer_nocopy(int64_t sess_id, int64_t host_ptr, int64_t byte_len) {
    @autoreleasepool {
        if (host_ptr <= 0 || byte_len <= 0) {
            fail("<nocopy>", "invalid-args");
            return 0;
        }
        const size_t page_sz = static_cast<size_t>(getpagesize());
        const uintptr_t addr = static_cast<uintptr_t>(host_ptr);
        if ((addr % page_sz) != 0) {
            fail("<nocopy>", "alignment");
            return 0;
        }
        const size_t raw_len = static_cast<size_t>(byte_len);
        const size_t aligned_len = (raw_len + page_sz - 1) & ~(page_sz - 1);
        id<MTLDevice> device = nil;
        {
            auto& st = bridge_store();
            std::lock_guard<std::mutex> lock(st.mu);
            auto it = st.sessions.find(sess_id);
            if (it == st.sessions.end()) return 0;
            device = it->second.device;
        }
        void* ptr = reinterpret_cast<void*>(addr);
        id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:ptr
                                                      length:aligned_len
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
        if (!buf) {
            fail("<nocopy>", "buffer");
            return 0;
        }
        if ([buf contents] != ptr) {
            fail("<nocopy>", "contents-mismatch");
            return 0;
        }
        auto& st = bridge_store();
        std::lock_guard<std::mutex> lock(st.mu);
        int64_t bid = st.next_id++;
        st.buffers.emplace(bid, MetalBufferState{sess_id, buf, ptr});
        return bid;
    }
}

extern "C" int64_t sounio_metal_buffer_alloc(int64_t sess_id, int64_t byte_len) {
    @autoreleasepool {
        if (byte_len <= 0) return 0;
        id<MTLDevice> device = nil;
        {
            auto& st = bridge_store();
            std::lock_guard<std::mutex> lock(st.mu);
            auto it = st.sessions.find(sess_id);
            if (it == st.sessions.end()) return 0;
            device = it->second.device;
        }
        id<MTLBuffer> buf = [device newBufferWithLength:static_cast<size_t>(byte_len)
                                                options:MTLResourceStorageModeShared];
        if (!buf) {
            fail("<alloc>", "buffer");
            return 0;
        }
        void* ptr = [buf contents];
        auto& st = bridge_store();
        std::lock_guard<std::mutex> lock(st.mu);
        int64_t bid = st.next_id++;
        st.buffers.emplace(bid, MetalBufferState{sess_id, buf, ptr});
        return bid;
    }
}

extern "C" int64_t sounio_metal_buffer_contents(int64_t buf_id) {
    auto& st = bridge_store();
    std::lock_guard<std::mutex> lock(st.mu);
    auto it = st.buffers.find(buf_id);
    if (it == st.buffers.end()) return 0;
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(it->second.raw_ptr));
}

extern "C" int64_t sounio_metal_buffer_free(int64_t buf_id) {
    @autoreleasepool {
        auto& st = bridge_store();
        std::lock_guard<std::mutex> lock(st.mu);
        st.buffers.erase(buf_id);
        return 0;
    }
}

extern "C" int64_t sounio_metal_compile_msl(int64_t sess_id, const char* msl_source, const char* fn_name) {
    @autoreleasepool {
        if (!msl_source || !fn_name) {
            fail(fn_name, "abi");
            return 0;
        }
        id<MTLDevice> device = nil;
        {
            auto& st = bridge_store();
            std::lock_guard<std::mutex> lock(st.mu);
            auto it = st.sessions.find(sess_id);
            if (it == st.sessions.end()) return 0;
            device = it->second.device;
        }
        NSString* src = [NSString stringWithUTF8String:msl_source];
        NSString* entry = [NSString stringWithUTF8String:fn_name];
        if (!src || !entry) {
            fail(fn_name, "source-encoding");
            return 0;
        }
        MTLCompileOptions* options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        } else {
            options.fastMathEnabled = NO;
        }
        options.languageVersion = MTLLanguageVersion3_0;
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:src options:options error:&error];
        if (!library) {
            fail(fn_name, "compile", error);
            return 0;
        }
        id<MTLFunction> function = [library newFunctionWithName:entry];
        if (!function) {
            fail(fn_name, "function");
            return 0;
        }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            fail(fn_name, "pipeline", error);
            return 0;
        }
        auto& st = bridge_store();
        std::lock_guard<std::mutex> lock(st.mu);
        int64_t pid = st.next_id++;
        st.pipelines.emplace(pid, MetalPipelineState{sess_id, pipeline});
        return pid;
    }
}

extern "C" int64_t sounio_metal_cmd_begin(int64_t sess_id) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = nil;
        {
            auto& st = bridge_store();
            std::lock_guard<std::mutex> lock(st.mu);
            auto it = st.sessions.find(sess_id);
            if (it == st.sessions.end()) return 0;
            queue = it->second.queue;
        }
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = command ? [command computeCommandEncoder] : nil;
        if (!command || !encoder) {
            fail("<cmd>", "command");
            return 0;
        }
        auto& st = bridge_store();
        std::lock_guard<std::mutex> lock(st.mu);
        int64_t cid = st.next_id++;
        st.commands.emplace(cid, MetalCommandState{sess_id, command, encoder});
        return cid;
    }
}

extern "C" int64_t sounio_metal_cmd_set_buffer(int64_t cmd_id, int64_t slot_idx, int64_t buf_id, int64_t byte_offset) {
    @autoreleasepool {
        if (slot_idx < 0 || byte_offset < 0) return -1;
        auto& st = bridge_store();
        std::lock_guard<std::mutex> lock(st.mu);
        auto cit = st.commands.find(cmd_id);
        auto bit = st.buffers.find(buf_id);
        if (cit == st.commands.end() || bit == st.buffers.end()) return -1;
        [cit->second.encoder setBuffer:bit->second.buffer
                                offset:static_cast<NSUInteger>(byte_offset)
                               atIndex:static_cast<NSUInteger>(slot_idx)];
        return 0;
    }
}

extern "C" int64_t sounio_metal_cmd_set_u64(int64_t cmd_id, int64_t slot_idx, int64_t value) {
    @autoreleasepool {
        if (slot_idx < 0) return -1;
        auto& st = bridge_store();
        std::lock_guard<std::mutex> lock(st.mu);
        auto cit = st.commands.find(cmd_id);
        if (cit == st.commands.end()) return -1;
        uint64_t v = static_cast<uint64_t>(value);
        [cit->second.encoder setBytes:&v length:8 atIndex:static_cast<NSUInteger>(slot_idx)];
        return 0;
    }
}

extern "C" int64_t sounio_metal_cmd_dispatch_1d(int64_t cmd_id, int64_t pipe_id, int64_t total_threads, int64_t group_threads) {
    @autoreleasepool {
        if (total_threads <= 0 || group_threads <= 0) return -1;
        auto& st = bridge_store();
        std::lock_guard<std::mutex> lock(st.mu);
        auto cit = st.commands.find(cmd_id);
        auto pit = st.pipelines.find(pipe_id);
        if (cit == st.commands.end() || pit == st.pipelines.end()) return -1;
        id<MTLComputePipelineState> pipe = pit->second.pipeline;
        const NSUInteger max_tg = pipe.maxTotalThreadsPerThreadgroup;
        const NSUInteger width = std::min<NSUInteger>(max_tg, static_cast<NSUInteger>(group_threads));
        if (width == 0) return -1;
        [cit->second.encoder setComputePipelineState:pipe];
        [cit->second.encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(total_threads), 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        return 0;
    }
}

extern "C" int64_t sounio_metal_cmd_commit_sync(int64_t cmd_id) {
    @autoreleasepool {
        id<MTLCommandBuffer> command = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        {
            auto& st = bridge_store();
            std::lock_guard<std::mutex> lock(st.mu);
            auto cit = st.commands.find(cmd_id);
            if (cit == st.commands.end()) return -1;
            command = cit->second.command;
            encoder = cit->second.encoder;
            st.commands.erase(cit);
        }
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            fail("<cmd>", "execute", command.error);
            return -1;
        }
        return 0;
    }
}

extern "C" int64_t sounio_metal_cmd_cancel(int64_t cmd_id) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = nil;
        {
            auto& st = bridge_store();
            std::lock_guard<std::mutex> lock(st.mu);
            auto cit = st.commands.find(cmd_id);
            if (cit == st.commands.end()) return -1;
            encoder = cit->second.encoder;
            st.commands.erase(cit);
        }
        if (encoder) {
            [encoder endEncoding];
        }
        return 0;
    }
}
