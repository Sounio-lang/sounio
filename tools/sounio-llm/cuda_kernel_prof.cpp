// ADR-009 C++23 CUDA Driver API per-kernel profiler shim (loaded via SOUNIO_CUDA_DRIVER_LIB).
// Forwards all cu* symbols used by stdlib/llm/hal.sio to real libcuda.so.1 while timing
// cuLaunchKernel calls via hardware cuEventCreate/cuEventRecord/cuEventElapsedTime.

#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUevent = void*;
using CUdeviceptr = unsigned long long;

struct KernelStat {
  uint64_t calls = 0;
  double total_ms = 0.0;
  double min_ms = 1e30;
  double max_ms = 0.0;
};

struct ProfState {
  void* real_lib = nullptr;
  CUresult (*fn_init)(unsigned int) = nullptr;
  CUresult (*fn_dev_get)(CUdevice*, int) = nullptr;
  CUresult (*fn_ctx_create)(CUcontext*, unsigned int, CUdevice) = nullptr;
  CUresult (*fn_ctx_destroy)(CUcontext) = nullptr;
  CUresult (*fn_mod_load)(CUmodule*, const void*) = nullptr;
  CUresult (*fn_mod_load_ex)(CUmodule*, const void*, unsigned int, int*, void**) = nullptr;
  CUresult (*fn_mod_unload)(CUmodule) = nullptr;
  CUresult (*fn_mod_get_fn)(CUfunction*, CUmodule, const char*) = nullptr;
  CUresult (*fn_mod_get_global)(CUdeviceptr*, size_t*, CUmodule, const char*) = nullptr;
  CUresult (*fn_mem_alloc)(CUdeviceptr*, size_t) = nullptr;
  CUresult (*fn_mem_free)(CUdeviceptr) = nullptr;
  CUresult (*fn_h2d)(CUdeviceptr, const void*, size_t) = nullptr;
  CUresult (*fn_d2h)(void*, CUdeviceptr, size_t) = nullptr;
  CUresult (*fn_launch)(CUfunction, unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int, unsigned int,
                        CUstream, void**, void**) = nullptr;
  CUresult (*fn_ctx_sync)() = nullptr;
  CUresult (*fn_dev_name)(char*, int, CUdevice) = nullptr;
  CUresult (*fn_drv_ver)(int*) = nullptr;
  CUresult (*fn_dev_attr)(int*, int, CUdevice) = nullptr;
  CUresult (*fn_ev_create)(CUevent*, unsigned int) = nullptr;
  CUresult (*fn_ev_destroy)(CUevent) = nullptr;
  CUresult (*fn_ev_record)(CUevent, CUstream) = nullptr;
  CUresult (*fn_ev_sync)(CUevent) = nullptr;
  CUresult (*fn_ev_elapsed)(float*, CUevent, CUevent) = nullptr;

  CUevent ev_start = nullptr;
  CUevent ev_stop = nullptr;
  std::mutex mtx;
  std::unordered_map<void*, std::string> fn_names;
  std::unordered_map<std::string, KernelStat> stats;
  bool printed = false;

  void ensure_loaded() {
    if (real_lib) return;
    real_lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!real_lib) {
      std::cerr << "cuda_kernel_prof: failed to dlopen libcuda.so.1: " << dlerror() << "\n";
      std::abort();
    }
    auto load = [&](auto& dst, const char* sym) {
      dst = reinterpret_cast<std::remove_reference_t<decltype(dst)>>(dlsym(real_lib, sym));
    };
    load(fn_init, "cuInit");
    load(fn_dev_get, "cuDeviceGet");
    load(fn_ctx_create, "cuCtxCreate_v2");
    load(fn_ctx_destroy, "cuCtxDestroy_v2");
    load(fn_mod_load, "cuModuleLoadData");
    load(fn_mod_load_ex, "cuModuleLoadDataEx");
    load(fn_mod_unload, "cuModuleUnload");
    load(fn_mod_get_fn, "cuModuleGetFunction");
    load(fn_mod_get_global, "cuModuleGetGlobal_v2");
    if (!fn_mod_get_global) load(fn_mod_get_global, "cuModuleGetGlobal");
    load(fn_mem_alloc, "cuMemAlloc_v2");
    load(fn_mem_free, "cuMemFree_v2");
    load(fn_h2d, "cuMemcpyHtoD_v2");
    load(fn_d2h, "cuMemcpyDtoH_v2");
    load(fn_launch, "cuLaunchKernel");
    load(fn_ctx_sync, "cuCtxSynchronize");
    load(fn_dev_name, "cuDeviceGetName");
    load(fn_drv_ver, "cuDriverGetVersion");
    load(fn_dev_attr, "cuDeviceGetAttribute");
    load(fn_ev_create, "cuEventCreate");
    load(fn_ev_destroy, "cuEventDestroy_v2");
    if (!fn_ev_destroy) load(fn_ev_destroy, "cuEventDestroy");
    load(fn_ev_record, "cuEventRecord");
    load(fn_ev_sync, "cuEventSynchronize");
    load(fn_ev_elapsed, "cuEventElapsedTime");
  }

  void dump_report() {
    std::lock_guard<std::mutex> lk(mtx);
    if (printed || stats.empty()) return;
    printed = true;

    std::vector<std::pair<std::string, KernelStat>> rows(stats.begin(), stats.end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
      return a.second.total_ms > b.second.total_ms;
    });

    double total_all_ms = 0.0;
    double total_decode_ms = 0.0;
    for (const auto& [name, st] : rows) {
      total_all_ms += st.total_ms;
      if (name.find("quantize") == std::string::npos) total_decode_ms += st.total_ms;
    }

    std::cerr << "\n=== CUDA_KERNEL_PROFILE (hardware cuEventElapsedTime) ===\n";
    std::cerr << std::left << std::setw(32) << "kernel" << std::right << std::setw(9) << "calls"
              << std::setw(12) << "total_ms" << std::setw(10) << "pct_all" << std::setw(11)
              << "pct_infer" << std::setw(11) << "mean_us" << "\n";
    std::cerr << std::fixed;
    for (const auto& [name, st] : rows) {
      const double pct_all = total_all_ms > 0.0 ? (100.0 * st.total_ms / total_all_ms) : 0.0;
      const bool is_quant = (name.find("quantize") != std::string::npos);
      const double pct_inf = (!is_quant && total_decode_ms > 0.0)
                                 ? (100.0 * st.total_ms / total_decode_ms)
                                 : 0.0;
      const double mean_us = st.calls > 0 ? (1000.0 * st.total_ms / static_cast<double>(st.calls)) : 0.0;
      std::cerr << std::left << std::setw(32) << name << std::right << std::setw(9) << st.calls
                << std::setw(12) << std::setprecision(3) << st.total_ms
                << std::setw(9) << std::setprecision(2) << pct_all << "%"
                << std::setw(10) << std::setprecision(2) << pct_inf << "%"
                << std::setw(11) << std::setprecision(1) << mean_us << "\n";
    }
    std::cerr << "TOTAL_GPU_ALL_MS=" << std::setprecision(3) << total_all_ms
              << " TOTAL_GPU_INFER_MS=" << total_decode_ms << "\n";
  }
};

ProfState& prof() {
  static ProfState s;
  s.ensure_loaded();
  return s;
}

}  // namespace

extern "C" {

CUresult cuInit(unsigned int flags) { return prof().fn_init(flags); }
CUresult cuDeviceGet(CUdevice* dev, int ord) { return prof().fn_dev_get(dev, ord); }

CUresult cuCtxCreate_v2(CUcontext* ctx, unsigned int flags, CUdevice dev) {
  auto& p = prof();
  CUresult rc = p.fn_ctx_create(ctx, flags, dev);
  if (rc == 0 && p.fn_ev_create && !p.ev_start) {
    p.fn_ev_create(&p.ev_start, 0);
    p.fn_ev_create(&p.ev_stop, 0);
  }
  return rc;
}

CUresult cuCtxDestroy_v2(CUcontext ctx) {
  auto& p = prof();
  p.dump_report();
  if (p.ev_start && p.fn_ev_destroy) {
    p.fn_ev_destroy(p.ev_start);
    p.fn_ev_destroy(p.ev_stop);
    p.ev_start = nullptr;
    p.ev_stop = nullptr;
  }
  return p.fn_ctx_destroy(ctx);
}

CUresult cuModuleLoadData(CUmodule* mod, const void* img) { return prof().fn_mod_load(mod, img); }
CUresult cuModuleLoadDataEx(CUmodule* mod, const void* img, unsigned int n, int* opts, void** vals) {
  return prof().fn_mod_load_ex(mod, img, n, opts, vals);
}
CUresult cuModuleUnload(CUmodule mod) { return prof().fn_mod_unload(mod); }

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
  auto& p = prof();
  CUresult rc = p.fn_mod_get_fn(hfunc, hmod, name);
  if (rc == 0 && hfunc && *hfunc && name) {
    std::lock_guard<std::mutex> lk(p.mtx);
    p.fn_names[*hfunc] = name;
  }
  return rc;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name) {
  return prof().fn_mod_get_global(dptr, bytes, hmod, name);
}
CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name) {
  return prof().fn_mod_get_global(dptr, bytes, hmod, name);
}
CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytes) { return prof().fn_mem_alloc(dptr, bytes); }
CUresult cuMemFree_v2(CUdeviceptr dptr) { return prof().fn_mem_free(dptr); }
CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void* src, size_t bytes) {
  return prof().fn_h2d(dst, src, bytes);
}
CUresult cuMemcpyDtoH_v2(void* dst, CUdeviceptr src, size_t bytes) {
  return prof().fn_d2h(dst, src, bytes);
}
CUresult cuCtxSynchronize() { return prof().fn_ctx_sync(); }
CUresult cuDeviceGetName(char* name, int len, CUdevice dev) { return prof().fn_dev_name(name, len, dev); }
CUresult cuDriverGetVersion(int* ver) { return prof().fn_drv_ver(ver); }
CUresult cuDeviceGetAttribute(int* pi, int attr, CUdevice dev) { return prof().fn_dev_attr(pi, attr, dev); }

CUresult cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz,
                        unsigned int bx, unsigned int by, unsigned int bz, unsigned int smem,
                        CUstream stream, void** params, void** extra) {
  auto& p = prof();
  if (!p.ev_start || !p.ev_stop) {
    return p.fn_launch(f, gx, gy, gz, bx, by, bz, smem, stream, params, extra);
  }
  p.fn_ev_record(p.ev_start, stream);
  CUresult rc = p.fn_launch(f, gx, gy, gz, bx, by, bz, smem, stream, params, extra);
  p.fn_ev_record(p.ev_stop, stream);
  if (rc == 0) {
    p.fn_ev_sync(p.ev_stop);
    float ms = 0.0f;
    if (p.fn_ev_elapsed(&ms, p.ev_start, p.ev_stop) == 0) {
      std::lock_guard<std::mutex> lk(p.mtx);
      auto it = p.fn_names.find(f);
      const std::string& kname = (it != p.fn_names.end()) ? it->second : std::string("<unknown>");
      auto& st = p.stats[kname];
      ++st.calls;
      st.total_ms += static_cast<double>(ms);
      if (ms < st.min_ms) st.min_ms = ms;
      if (ms > st.max_ms) st.max_ms = ms;
    }
  }
  return rc;
}

}  // extern "C"
