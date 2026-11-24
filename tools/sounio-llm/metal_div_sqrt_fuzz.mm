// ADR-009 verified_foreign_reference (Objective-C++ / C++23 Metal vs CPU RNE FTZ+DAZ fuzzer)
// 1. Exhaustive 2^32 binary32 sqrt on Metal default sqrt, precise::sqrt, and fast::sqrt vs CPU RNE (FTZ+DAZ).
// 2. 1e9 random (a, b) pairs over the full binary32 exponent range + 1e6 directed rounding ties +
//    1e6 subnormal-boundary quotients for Metal default '/', precise::divide, and fast::divide vs CPU RNE (FTZ+DAZ).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

static inline void cpu_enable_ftz() {
#if defined(__aarch64__)
  uint64_t fpcr = 0;
  asm volatile("mrs %0, fpcr" : "=r"(fpcr));
  fpcr |= (1ULL << 24);  // FZ: flush-to-zero (both inputs and outputs on AArch64)
  asm volatile("msr fpcr, %0" ::"r"(fpcr));
#endif
}

static inline uint32_t cpu_sqrt_ftz_bits(uint32_t u) {
  float x = std::bit_cast<float>(u);
  float r;
#if defined(__aarch64__)
  asm volatile("fsqrt %s0, %s1" : "=w"(r) : "w"(x));
#else
  r = std::sqrt(x);
#endif
  return std::bit_cast<uint32_t>(r);
}

static inline uint32_t cpu_div_ftz_bits(uint32_t ua, uint32_t ub) {
  float a = std::bit_cast<float>(ua);
  float b = std::bit_cast<float>(ub);
  float r;
#if defined(__aarch64__)
  asm volatile("fdiv %s0, %s1, %s2" : "=w"(r) : "w"(a), "w"(b));
#else
  r = a / b;
#endif
  return std::bit_cast<uint32_t>(r);
}

static inline bool is_nan_u32(uint32_t u) {
  return (u & 0x7fffffffu) > 0x7f800000u;
}

static inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

static inline void gen_div_random_pair(uint64_t idx, uint32_t& ua, uint32_t& ub) {
  uint64_t h = splitmix64(idx ^ 0x243f6a8885a308d3ULL);
  ua = static_cast<uint32_t>(h & 0xffffffffu);
  ub = static_cast<uint32_t>(h >> 32);
}

static inline void gen_div_tie_pair(uint64_t idx, uint32_t& ua, uint32_t& ub) {
  uint64_t h = splitmix64(idx ^ 0x13198a2e03707344ULL);
  uint32_t sign_a = static_cast<uint32_t>((h & 1ULL) << 31);
  uint32_t sign_b = static_cast<uint32_t>(((h >> 1) & 1ULL) << 31);
  uint64_t q_half = (1ULL << 24) + 2ULL * ((h >> 2) & 0x3fffffULL) + 1ULL;
  uint64_t m_b = (1ULL << 23) + ((h >> 24) & 0x7fffffULL);
  uint64_t prod = q_half * m_b;
  uint64_t m_a = (prod + (1ULL << 23)) >> 24;
  int exp_a = 127 + static_cast<int>(((h >> 47) % 101ULL)) - 50;
  int exp_b = 127 + static_cast<int>(((h >> 55) % 81ULL)) - 40;
  if (m_a >= (1ULL << 24)) {
    m_a >>= 1;
    exp_a += 1;
  }
  ua = sign_a | (static_cast<uint32_t>(exp_a) << 23) | static_cast<uint32_t>(m_a & 0x7fffffULL);
  ub = sign_b | (static_cast<uint32_t>(exp_b) << 23) | static_cast<uint32_t>(m_b & 0x7fffffULL);
}

static inline void gen_div_subnorm_pair(uint64_t idx, uint32_t& ua, uint32_t& ub) {
  uint64_t h = splitmix64(idx ^ 0xa4093822299f31d0ULL);
  uint32_t sign = static_cast<uint32_t>((h & 1ULL) << 31);
  uint32_t mode = static_cast<uint32_t>((h >> 1) & 3ULL);
  uint32_t m = static_cast<uint32_t>((h >> 3) & 0x7ffffcULL) + 2u;
  if (mode == 0) {
    ua = sign | (0x00800000u | m);
    ub = 0x3f800000u | (m + 1u);
  } else if (mode == 1) {
    ua = sign | (0x00800000u | (m + 1u));
    ub = 0x3f800000u | m;
  } else if (mode == 2) {
    int delta = static_cast<int>((h >> 27) & 127ULL) - 64;
    ua = sign | static_cast<uint32_t>(0x01000000 + delta);
    ub = 0x40000000u + static_cast<uint32_t>(((h >> 34) & 31ULL));
  } else {
    ua = sign | 0x00800000u;
    ub = 0x3f800000u + static_cast<uint32_t>(((h >> 27) & 0x7fffffu));
  }
}

static const char* kMslSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

inline ulong splitmix64(ulong x) {
    x += 0x9e3779b97f4a7c15ul;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ul;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebul;
    return x ^ (x >> 31);
}

inline void gen_div_pair(ulong idx, uint kind, thread uint& ua, thread uint& ub) {
    if (kind == 0u) {
        ulong h = splitmix64(idx ^ 0x243f6a8885a308d3ul);
        ua = (uint)(h & 0xfffffffful);
        ub = (uint)(h >> 32);
    } else if (kind == 1u) {
        ulong h = splitmix64(idx ^ 0x13198a2e03707344ul);
        uint sign_a = (uint)((h & 1ul) << 31);
        uint sign_b = (uint)(((h >> 1) & 1ul) << 31);
        ulong q_half = (1ul << 24) + 2ul * ((h >> 2) & 0x3ffffful) + 1ul;
        ulong m_b = (1ul << 23) + ((h >> 24) & 0x7ffffful);
        ulong prod = q_half * m_b;
        ulong m_a = (prod + (1ul << 23)) >> 24;
        int exp_a = 127 + (int)((h >> 47) % 101ul) - 50;
        int exp_b = 127 + (int)((h >> 55) % 81ul) - 40;
        if (m_a >= (1ul << 24)) {
            m_a >>= 1;
            exp_a += 1;
        }
        ua = sign_a | ((uint)exp_a << 23) | (uint)(m_a & 0x7ffffful);
        ub = sign_b | ((uint)exp_b << 23) | (uint)(m_b & 0x7ffffful);
    } else {
        ulong h = splitmix64(idx ^ 0xa4093822299f31d0ul);
        uint sign = (uint)((h & 1ul) << 31);
        uint mode = (uint)((h >> 1) & 3ul);
        uint m = (uint)((h >> 3) & 0x7ffffcul) + 2u;
        if (mode == 0u) {
            ua = sign | (0x00800000u | m);
            ub = 0x3f800000u | (m + 1u);
        } else if (mode == 1u) {
            ua = sign | (0x00800000u | (m + 1u));
            ub = 0x3f800000u | m;
        } else if (mode == 2u) {
            int delta = (int)((h >> 27) & 127ul) - 64;
            ua = sign | (uint)(0x01000000 + delta);
            ub = 0x40000000u + (uint)((h >> 34) & 31ul);
        } else {
            ua = sign | 0x00800000u;
            ub = 0x3f800000u + (uint)((h >> 27) & 0x7fffffu);
        }
    }
}

kernel void fuzz_sqrt_default(device uint* out [[buffer(0)]], constant ulong& base [[buffer(1)]],
                              constant uint& count [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint u = (uint)(base + (ulong)gid);
    out[gid] = as_type<uint>(sqrt(as_type<float>(u)));
}

kernel void fuzz_sqrt_precise(device uint* out [[buffer(0)]], constant ulong& base [[buffer(1)]],
                              constant uint& count [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint u = (uint)(base + (ulong)gid);
    out[gid] = as_type<uint>(precise::sqrt(as_type<float>(u)));
}

kernel void fuzz_sqrt_fast(device uint* out [[buffer(0)]], constant ulong& base [[buffer(1)]],
                           constant uint& count [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint u = (uint)(base + (ulong)gid);
    out[gid] = as_type<uint>(fast::sqrt(as_type<float>(u)));
}

kernel void fuzz_div_default(device uint* out [[buffer(0)]], constant ulong& base [[buffer(1)]],
                             constant uint& count [[buffer(2)]], constant uint& kind [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint ua, ub;
    gen_div_pair(base + (ulong)gid, kind, ua, ub);
    out[gid] = as_type<uint>(as_type<float>(ua) / as_type<float>(ub));
}

kernel void fuzz_div_precise(device uint* out [[buffer(0)]], constant ulong& base [[buffer(1)]],
                             constant uint& count [[buffer(2)]], constant uint& kind [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint ua, ub;
    gen_div_pair(base + (ulong)gid, kind, ua, ub);
    out[gid] = as_type<uint>(precise::divide(as_type<float>(ua), as_type<float>(ub)));
}

kernel void fuzz_div_fast(device uint* out [[buffer(0)]], constant ulong& base [[buffer(1)]],
                          constant uint& count [[buffer(2)]], constant uint& kind [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint ua, ub;
    gen_div_pair(base + (ulong)gid, kind, ua, ub);
    out[gid] = as_type<uint>(fast::divide(as_type<float>(ua), as_type<float>(ub)));
}
)MSL";

struct SqrtMismatch {
  uint32_t in_u;
  uint32_t gpu_u;
  uint32_t cpu_u;
};

struct DivMismatch {
  uint32_t a_u;
  uint32_t b_u;
  uint32_t gpu_u;
  uint32_t cpu_u;
};

static id<MTLComputePipelineState> make_pipe(id<MTLDevice> dev, id<MTLLibrary> lib, const char* name) {
  NSError* err = nil;
  id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
  if (!fn) {
    std::cerr << "Missing Metal function: " << name << "\n";
    std::exit(1);
  }
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
  if (!pso) {
    std::cerr << "Pipeline error for " << name << ": " << [[err localizedDescription] UTF8String] << "\n";
    std::exit(1);
  }
  return pso;
}

static void run_sqrt_exhaustive(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLComputePipelineState> pso,
                                const char* label, id<MTLBuffer> buf, uint32_t chunk_n) {
  auto t0 = std::chrono::steady_clock::now();
  uint32_t* ptr = static_cast<uint32_t*>([buf contents]);
  const unsigned n_threads = std::max(1u, std::thread::hardware_concurrency());
  std::atomic<uint64_t> non_nan_mism{0};
  std::atomic<uint64_t> nan_payload_diff{0};
  std::mutex mtx;
  std::vector<SqrtMismatch> samples;

  const uint64_t total = 1ULL << 32;
  for (uint64_t base = 0; base < total; base += chunk_n) {
    uint32_t cnt = static_cast<uint32_t>(std::min<uint64_t>(chunk_n, total - base));
    @autoreleasepool {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:buf offset:0 atIndex:0];
      [enc setBytes:&base length:sizeof(base) atIndex:1];
      [enc setBytes:&cnt length:sizeof(cnt) atIndex:2];
      MTLSize tg = MTLSizeMake(256, 1, 1);
      MTLSize grid = MTLSizeMake(cnt, 1, 1);
      [enc dispatchThreads:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    }

    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    const uint32_t per = (cnt + n_threads - 1) / n_threads;
    for (unsigned t = 0; t < n_threads; ++t) {
      uint32_t lo = t * per;
      uint32_t hi = std::min(cnt, lo + per);
      if (lo >= hi) continue;
      workers.emplace_back([&, lo, hi, base]() {
        cpu_enable_ftz();
        uint64_t local_mism = 0;
        uint64_t local_nan = 0;
        std::vector<SqrtMismatch> local_s;
        for (uint32_t i = lo; i < hi; ++i) {
          uint32_t u = static_cast<uint32_t>(base + i);
          uint32_t g = ptr[i];
          uint32_t c = cpu_sqrt_ftz_bits(u);
          if (g != c) {
            if (is_nan_u32(g) && is_nan_u32(c)) {
              ++local_nan;
            } else {
              ++local_mism;
              if (local_s.size() < 10) local_s.push_back({u, g, c});
            }
          }
        }
        non_nan_mism.fetch_add(local_mism, std::memory_order_relaxed);
        nan_payload_diff.fetch_add(local_nan, std::memory_order_relaxed);
        if (!local_s.empty()) {
          std::lock_guard<std::mutex> lk(mtx);
          for (const auto& s : local_s) {
            if (samples.size() < 10) samples.push_back(s);
          }
        }
      });
    }
    for (auto& w : workers) w.join();
  }

  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "SQRT_EXHAUSTIVE [" << label << "] total=4294967296 non_nan_mismatches="
            << non_nan_mism.load() << " nan_payload_diffs=" << nan_payload_diff.load()
            << " elapsed_s=" << std::fixed << std::setprecision(2) << secs << "\n";
  for (size_t i = 0; i < samples.size(); ++i) {
    const auto& s = samples[i];
    std::printf("  offend[%zu]: in=0x%08x (%.9g) metal=0x%08x (%.9g) cpu=0x%08x (%.9g)\n",
                i, s.in_u, std::bit_cast<float>(s.in_u),
                s.gpu_u, std::bit_cast<float>(s.gpu_u),
                s.cpu_u, std::bit_cast<float>(s.cpu_u));
  }
}

static void run_div_fuzz(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLComputePipelineState> pso,
                         const char* label, uint32_t kind, uint64_t total, id<MTLBuffer> buf,
                         uint32_t chunk_n) {
  auto t0 = std::chrono::steady_clock::now();
  uint32_t* ptr = static_cast<uint32_t*>([buf contents]);
  const unsigned n_threads = std::max(1u, std::thread::hardware_concurrency());
  std::atomic<uint64_t> non_nan_mism{0};
  std::atomic<uint64_t> nan_payload_diff{0};
  std::mutex mtx;
  std::vector<DivMismatch> samples;

  for (uint64_t base = 0; base < total; base += chunk_n) {
    uint32_t cnt = static_cast<uint32_t>(std::min<uint64_t>(chunk_n, total - base));
    @autoreleasepool {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:buf offset:0 atIndex:0];
      [enc setBytes:&base length:sizeof(base) atIndex:1];
      [enc setBytes:&cnt length:sizeof(cnt) atIndex:2];
      [enc setBytes:&kind length:sizeof(kind) atIndex:3];
      MTLSize tg = MTLSizeMake(256, 1, 1);
      MTLSize grid = MTLSizeMake(cnt, 1, 1);
      [enc dispatchThreads:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    }

    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    const uint32_t per = (cnt + n_threads - 1) / n_threads;
    for (unsigned t = 0; t < n_threads; ++t) {
      uint32_t lo = t * per;
      uint32_t hi = std::min(cnt, lo + per);
      if (lo >= hi) continue;
      workers.emplace_back([&, lo, hi, base]() {
        cpu_enable_ftz();
        uint64_t local_mism = 0;
        uint64_t local_nan = 0;
        std::vector<DivMismatch> local_s;
        for (uint32_t i = lo; i < hi; ++i) {
          uint32_t ua = 0, ub = 0;
          if (kind == 0u) gen_div_random_pair(base + i, ua, ub);
          else if (kind == 1u) gen_div_tie_pair(base + i, ua, ub);
          else gen_div_subnorm_pair(base + i, ua, ub);
          uint32_t g = ptr[i];
          uint32_t c = cpu_div_ftz_bits(ua, ub);
          if (g != c) {
            if (is_nan_u32(g) && is_nan_u32(c)) {
              ++local_nan;
            } else {
              ++local_mism;
              if (local_s.size() < 10) local_s.push_back({ua, ub, g, c});
            }
          }
        }
        non_nan_mism.fetch_add(local_mism, std::memory_order_relaxed);
        nan_payload_diff.fetch_add(local_nan, std::memory_order_relaxed);
        if (!local_s.empty()) {
          std::lock_guard<std::mutex> lk(mtx);
          for (const auto& s : local_s) {
            if (samples.size() < 10) samples.push_back(s);
          }
        }
      });
    }
    for (auto& w : workers) w.join();
  }

  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "DIV_FUZZ [" << label << "] total=" << total << " non_nan_mismatches="
            << non_nan_mism.load() << " nan_payload_diffs=" << nan_payload_diff.load()
            << " elapsed_s=" << std::fixed << std::setprecision(2) << secs << "\n";
  for (size_t i = 0; i < samples.size(); ++i) {
    const auto& s = samples[i];
    std::printf("  offend[%zu]: a=0x%08x (%.9g) b=0x%08x (%.9g) metal=0x%08x (%.9g) cpu=0x%08x (%.9g)\n",
                i, s.a_u, std::bit_cast<float>(s.a_u),
                s.b_u, std::bit_cast<float>(s.b_u),
                s.gpu_u, std::bit_cast<float>(s.gpu_u),
                s.cpu_u, std::bit_cast<float>(s.cpu_u));
  }
}

}  // namespace

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
      std::cerr << "No Metal device\n";
      return 1;
    }
    std::cout << "METAL_DEVICE=" << [[dev name] UTF8String]
              << " fastMathEnabled=NO threads=" << std::thread::hardware_concurrency() << "\n";
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.fastMathEnabled = NO;
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kMslSource]
                                           options:opts
                                             error:&err];
    if (!lib) {
      std::cerr << "MSL compile error: " << [[err localizedDescription] UTF8String] << "\n";
      return 1;
    }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    auto p_sq_def = make_pipe(dev, lib, "fuzz_sqrt_default");
    auto p_sq_pre = make_pipe(dev, lib, "fuzz_sqrt_precise");
    auto p_sq_fst = make_pipe(dev, lib, "fuzz_sqrt_fast");
    auto p_dv_def = make_pipe(dev, lib, "fuzz_div_default");
    auto p_dv_pre = make_pipe(dev, lib, "fuzz_div_precise");
    auto p_dv_fst = make_pipe(dev, lib, "fuzz_div_fast");

    const uint32_t chunk_n = 1u << 26;  // 67,108,864 floats (256 MiB)
    id<MTLBuffer> buf = [dev newBufferWithLength:static_cast<size_t>(chunk_n) * sizeof(uint32_t)
                                         options:MTLResourceStorageModeShared];

    run_sqrt_exhaustive(dev, q, p_sq_def, "metal_default_sqrt", buf, chunk_n);
    run_sqrt_exhaustive(dev, q, p_sq_pre, "metal_precise_sqrt", buf, chunk_n);
    run_sqrt_exhaustive(dev, q, p_sq_fst, "metal_fast_sqrt_posctrl", buf, chunk_n);

    run_div_fuzz(dev, q, p_dv_def, "metal_default_div_random_1e9", 0u, 1000000000ULL, buf, chunk_n);
    run_div_fuzz(dev, q, p_dv_def, "metal_default_div_ties_1e6", 1u, 1000000ULL, buf, chunk_n);
    run_div_fuzz(dev, q, p_dv_def, "metal_default_div_subnorm_1e6", 2u, 1000000ULL, buf, chunk_n);

    run_div_fuzz(dev, q, p_dv_pre, "metal_precise_div_random_1e9", 0u, 1000000000ULL, buf, chunk_n);
    run_div_fuzz(dev, q, p_dv_pre, "metal_precise_div_ties_1e6", 1u, 1000000ULL, buf, chunk_n);
    run_div_fuzz(dev, q, p_dv_pre, "metal_precise_div_subnorm_1e6", 2u, 1000000ULL, buf, chunk_n);

    run_div_fuzz(dev, q, p_dv_fst, "metal_fast_div_ties_1e6_posctrl", 1u, 1000000ULL, buf, chunk_n);
  }
  return 0;
}
