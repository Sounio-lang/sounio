// C++ MATERIAL_PARITY probe for the frozen Sounio Apple CPU interface schema.
// It reports material facts and raw samples; Sounio alone classifies them.

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <libproc.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <string>
#include <sys/sysctl.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr char kSounioSourceSha256[] =
    "d8c7e6f9410c36f6858fb2379efa010a5adbaa32c615d89edc3e764a0606a6be";
constexpr char kSounioSemanticsSha256[] =
    "6819916ac4240923a149dd95ee9dcbeaba8d3826b7452dd819e177ff62ce8c7f";
constexpr char kAuthorityCommit[] =
    "ba85ed0689484f747e392783de4f912001153360";
constexpr std::uint32_t kKpcClassFixedMask = 1U;
constexpr int kProcPidThreadCounts = 34;
constexpr std::size_t kWarmupCount = 128;
constexpr std::size_t kSampleCount = 1001;
constexpr std::size_t kMaximumKpcCounters = 64;

struct ProcThreadCountsData {
  std::uint64_t instructions;
  std::uint64_t cycles;
  std::uint64_t user_time_mach;
  std::uint64_t system_time_mach;
  std::uint64_t energy_nj;
};

struct ProcThreadCounts {
  std::uint16_t length;
  std::uint16_t reserved0;
  std::uint32_t reserved1;
  ProcThreadCountsData counts[];
};

using KpcGetCounterCount = std::uint32_t (*)(std::uint32_t);
using KpcGetConfigCount = std::uint32_t (*)(std::uint32_t);
using KpcGetCurcpuCounters = int (*)(std::uint32_t, int *, std::uint64_t *);
using KpcGetFixedConfig = int (*)(std::uint64_t *);
using KpcGetPmuVersion = int (*)();
using KpcGetClasses = std::uint32_t (*)();
using KpcGetRunning = std::uint32_t (*)();
using KpcGetForceAllCounters = int (*)();

struct KpcApi {
  void *handle = nullptr;
  std::string image = "ABSENT";
  KpcGetCounterCount get_counter_count = nullptr;
  KpcGetConfigCount get_config_count = nullptr;
  KpcGetCurcpuCounters get_curcpu_counters = nullptr;
  KpcGetFixedConfig get_fixed_config = nullptr;
  KpcGetPmuVersion get_pmu_version = nullptr;
  KpcGetClasses get_classes = nullptr;
  KpcGetRunning get_running = nullptr;
  KpcGetForceAllCounters get_force_all_counters = nullptr;
};

struct ThreadCountsRead {
  int copied = -1;
  int error = 0;
  std::uint16_t returned_levels = 0;
  std::uint64_t instructions = 0;
  std::uint64_t cycles = 0;
};

std::string sysctl_string(const char *name) {
  std::size_t size = 0;
  if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) {
    return "UNAVAILABLE";
  }
  std::vector<char> value(size, '\0');
  if (sysctlbyname(name, value.data(), &size, nullptr, 0) != 0) {
    return "UNAVAILABLE";
  }
  if (size > 0 && value[size - 1] == '\0') {
    --size;
  }
  return std::string(value.data(), size);
}

template <typename T>
bool sysctl_scalar(const char *name, T *value, int *error) {
  std::size_t size = sizeof(*value);
  errno = 0;
  if (sysctlbyname(name, value, &size, nullptr, 0) != 0 ||
      size != sizeof(*value)) {
    *error = errno;
    return false;
  }
  *error = 0;
  return true;
}

template <typename Function>
Function symbol(void *handle, const char *name) {
  return reinterpret_cast<Function>(dlsym(handle, name));
}

KpcApi load_kpc() {
  constexpr std::array<const char *, 2> images = {
      "/System/Library/PrivateFrameworks/kperf.framework/Versions/A/kperf",
      "/System/Library/PrivateFrameworks/kperf.framework/kperf",
  };
  KpcApi api;
  for (const char *image : images) {
    api.handle = dlopen(image, RTLD_LAZY | RTLD_LOCAL);
    if (api.handle != nullptr) {
      api.image = image;
      break;
    }
  }
  if (api.handle == nullptr) {
    return api;
  }
  api.get_counter_count =
      symbol<KpcGetCounterCount>(api.handle, "kpc_get_counter_count");
  api.get_config_count =
      symbol<KpcGetConfigCount>(api.handle, "kpc_get_config_count");
  api.get_curcpu_counters =
      symbol<KpcGetCurcpuCounters>(api.handle, "kpc_get_curcpu_counters");
  api.get_fixed_config =
      symbol<KpcGetFixedConfig>(api.handle, "kpc_get_fixed_config");
  api.get_pmu_version =
      symbol<KpcGetPmuVersion>(api.handle, "kpc_get_pmu_version");
  api.get_classes = symbol<KpcGetClasses>(api.handle, "kpc_get_classes");
  api.get_running = symbol<KpcGetRunning>(api.handle, "kpc_get_running");
  api.get_force_all_counters = symbol<KpcGetForceAllCounters>(
      api.handle, "kpc_get_force_all_ctrs");
  return api;
}

__attribute__((noinline)) std::uint64_t dependency_chain(std::uint64_t value) {
  for (std::uint64_t i = 0; i < 64; ++i) {
    value = value * 6364136223846793005ULL + 1442695040888963407ULL;
    asm volatile("" : "+r"(value) : : "memory");
  }
  return value;
}

std::uint64_t median(std::vector<std::uint64_t> values) {
  if (values.empty()) {
    return 0;
  }
  const std::size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  return values[middle];
}

ThreadCountsRead read_thread_counts(std::uint64_t thread_id,
                                    std::uint32_t perf_levels,
                                    std::vector<std::uint8_t> *storage) {
  ThreadCountsRead read;
  std::fill(storage->begin(), storage->end(), 0);
  errno = 0;
  read.copied = proc_pidinfo(getpid(), kProcPidThreadCounts, thread_id,
                             storage->data(),
                             static_cast<int>(storage->size()));
  read.error = errno;
  if (read.copied < static_cast<int>(sizeof(ProcThreadCounts))) {
    return read;
  }
  const auto *counts =
      reinterpret_cast<const ProcThreadCounts *>(storage->data());
  read.returned_levels = counts->length;
  const std::size_t payload =
      static_cast<std::size_t>(read.copied) - sizeof(ProcThreadCounts);
  const std::size_t copied_levels = payload / sizeof(ProcThreadCountsData);
  const std::size_t levels = std::min<std::size_t>(
      {perf_levels, counts->length, copied_levels});
  for (std::size_t i = 0; i < levels; ++i) {
    read.instructions += counts->counts[i].instructions;
    read.cycles += counts->counts[i].cycles;
  }
  return read;
}

void emit_bool(const char *key, bool value) {
  std::cout << key << '=' << (value ? "true" : "false") << '\n';
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " RAW_SAMPLE_PATH\n";
    return 64;
  }
  std::ofstream raw(argv[1], std::ios::out | std::ios::trunc);
  if (!raw) {
    std::cerr << "unable to open raw sample path\n";
    return 65;
  }

  char hostname[256] = {};
  const int hostname_rc = gethostname(hostname, sizeof(hostname) - 1);
  struct utsname system_name {};
  const int uname_rc = uname(&system_name);

  std::uint32_t perf_levels = 0;
  int perf_levels_error = 0;
  const bool perf_levels_available =
      sysctl_scalar("hw.nperflevels", &perf_levels, &perf_levels_error) &&
      perf_levels > 0 && perf_levels <= 64;

  std::uint64_t cpu_frequency = 0;
  int cpu_frequency_error = 0;
  const bool cpu_frequency_available =
      sysctl_scalar("hw.cpufrequency", &cpu_frequency, &cpu_frequency_error);

  std::uint64_t thread_id = 0;
  const int thread_id_rc = pthread_threadid_np(nullptr, &thread_id);

  for (std::size_t i = 0; i < kWarmupCount; ++i) {
    (void)dependency_chain(i + 1);
  }

  KpcApi kpc = load_kpc();
  const bool kpc_symbols_complete =
      kpc.get_counter_count != nullptr &&
      kpc.get_curcpu_counters != nullptr;
  std::uint32_t kpc_counter_count = 0;
  std::uint32_t kpc_config_count = 0;
  if (kpc_symbols_complete) {
    kpc_counter_count = kpc.get_counter_count(kKpcClassFixedMask);
    if (kpc.get_config_count != nullptr) {
      kpc_config_count = kpc.get_config_count(kKpcClassFixedMask);
    }
  }
  const bool kpc_count_valid =
      kpc_counter_count > 0 && kpc_counter_count <= kMaximumKpcCounters;
  const bool kpc_config_count_valid =
      kpc_config_count > 0 && kpc_config_count <= kMaximumKpcCounters;

  int kpc_pmu_version = -1;
  std::uint32_t kpc_classes = 0;
  std::uint32_t kpc_running = 0;
  int kpc_force_all = -1;
  if (kpc.get_pmu_version != nullptr) {
    kpc_pmu_version = kpc.get_pmu_version();
  }
  if (kpc.get_classes != nullptr) {
    kpc_classes = kpc.get_classes();
  }
  if (kpc.get_running != nullptr) {
    kpc_running = kpc.get_running();
  }
  if (kpc.get_force_all_counters != nullptr) {
    kpc_force_all = kpc.get_force_all_counters();
  }

  int kpc_fixed_config_rc = -1;
  std::vector<std::uint64_t> kpc_fixed_config(kpc_config_count, 0);
  if (kpc_config_count_valid && kpc.get_fixed_config != nullptr) {
    kpc_fixed_config_rc = kpc.get_fixed_config(kpc_fixed_config.data());
  }

  std::vector<std::vector<std::uint64_t>> kpc_deltas(kpc_counter_count);
  std::vector<std::uint64_t> kpc_before(kpc_counter_count, 0);
  std::vector<std::uint64_t> kpc_after(kpc_counter_count, 0);
  std::size_t kpc_read_successes = 0;
  std::size_t kpc_accepted_samples = 0;
  std::size_t kpc_migrations = 0;
  int kpc_first_before_rc = -1;
  int kpc_first_after_rc = -1;
  int kpc_first_before_cpu = -1;
  int kpc_first_after_cpu = -1;

  std::vector<std::uint64_t> thread_cycle_deltas;
  std::vector<std::uint64_t> thread_instruction_deltas;
  std::size_t thread_read_successes = 0;
  int thread_first_before_rc = -1;
  int thread_first_after_rc = -1;
  int thread_first_before_error = 0;
  int thread_first_after_error = 0;
  std::uint16_t thread_first_returned_levels = 0;
  const std::size_t thread_buffer_size =
      sizeof(ProcThreadCounts) +
      static_cast<std::size_t>(perf_levels_available ? perf_levels : 1) *
          sizeof(ProcThreadCountsData);
  std::vector<std::uint8_t> thread_before_buffer(thread_buffer_size, 0);
  std::vector<std::uint8_t> thread_after_buffer(thread_buffer_size, 0);

  std::vector<std::uint64_t> absolute_time_deltas;
  std::vector<std::uint64_t> uptime_raw_deltas;
  std::size_t uptime_raw_successes = 0;
  std::uint64_t sink = 1;

  raw << "sample\tkpc_before_rc\tkpc_after_rc\tkpc_before_cpu"
         "\tkpc_after_cpu\tkpc_accepted\tthread_before_rc"
         "\tthread_after_rc\tthread_cycles_delta"
         "\tthread_instructions_delta\tabsolute_time_delta"
         "\tuptime_raw_delta";
  for (std::uint32_t i = 0; i < kpc_counter_count; ++i) {
    raw << "\tkpc_delta_" << i;
  }
  raw << '\n';

  for (std::size_t sample = 0; sample < kSampleCount; ++sample) {
    int before_cpu = -1;
    int after_cpu = -1;
    int before_rc = -1;
    int after_rc = -1;
    if (kpc_count_valid) {
      std::atomic_signal_fence(std::memory_order_seq_cst);
      before_rc = kpc.get_curcpu_counters(
          kKpcClassFixedMask, &before_cpu, kpc_before.data());
      std::atomic_signal_fence(std::memory_order_seq_cst);
    }

    const ThreadCountsRead thread_before =
        perf_levels_available && thread_id_rc == 0
            ? read_thread_counts(thread_id, perf_levels,
                                 &thread_before_buffer)
            : ThreadCountsRead{};
    const std::uint64_t absolute_before = mach_absolute_time();
    const std::uint64_t uptime_before =
        clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    sink = dependency_chain(sink + sample + 1);
    const std::uint64_t uptime_after =
        clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    const std::uint64_t absolute_after = mach_absolute_time();
    const ThreadCountsRead thread_after =
        perf_levels_available && thread_id_rc == 0
            ? read_thread_counts(thread_id, perf_levels,
                                 &thread_after_buffer)
            : ThreadCountsRead{};

    if (kpc_count_valid) {
      std::atomic_signal_fence(std::memory_order_seq_cst);
      after_rc = kpc.get_curcpu_counters(
          kKpcClassFixedMask, &after_cpu, kpc_after.data());
      std::atomic_signal_fence(std::memory_order_seq_cst);
    }

    if (sample == 0) {
      kpc_first_before_rc = before_rc;
      kpc_first_after_rc = after_rc;
      kpc_first_before_cpu = before_cpu;
      kpc_first_after_cpu = after_cpu;
      thread_first_before_rc = thread_before.copied;
      thread_first_after_rc = thread_after.copied;
      thread_first_before_error = thread_before.error;
      thread_first_after_error = thread_after.error;
      thread_first_returned_levels = thread_after.returned_levels;
    }

    const bool kpc_read_success = before_rc == 0 && after_rc == 0;
    const bool kpc_accepted = kpc_read_success && before_cpu == after_cpu;
    if (kpc_read_success) {
      ++kpc_read_successes;
      if (before_cpu != after_cpu) {
        ++kpc_migrations;
      }
    }
    if (kpc_accepted) {
      ++kpc_accepted_samples;
      for (std::uint32_t i = 0; i < kpc_counter_count; ++i) {
        kpc_deltas[i].push_back(kpc_after[i] - kpc_before[i]);
      }
    }

    const bool thread_success =
        thread_before.copied >= static_cast<int>(sizeof(ProcThreadCounts)) &&
        thread_after.copied >= static_cast<int>(sizeof(ProcThreadCounts));
    const std::uint64_t thread_cycles_delta =
        thread_success ? thread_after.cycles - thread_before.cycles : 0;
    const std::uint64_t thread_instructions_delta =
        thread_success
            ? thread_after.instructions - thread_before.instructions
            : 0;
    if (thread_success) {
      ++thread_read_successes;
      thread_cycle_deltas.push_back(thread_cycles_delta);
      thread_instruction_deltas.push_back(thread_instructions_delta);
    }

    const std::uint64_t absolute_delta = absolute_after - absolute_before;
    const std::uint64_t uptime_delta = uptime_after - uptime_before;
    absolute_time_deltas.push_back(absolute_delta);
    uptime_raw_deltas.push_back(uptime_delta);
    if (uptime_after >= uptime_before) {
      ++uptime_raw_successes;
    }

    raw << sample << '\t' << before_rc << '\t' << after_rc << '\t'
        << before_cpu << '\t' << after_cpu << '\t'
        << (kpc_accepted ? 1 : 0) << '\t' << thread_before.copied << '\t'
        << thread_after.copied << '\t' << thread_cycles_delta << '\t'
        << thread_instructions_delta << '\t' << absolute_delta << '\t'
        << uptime_delta;
    for (std::uint32_t i = 0; i < kpc_counter_count; ++i) {
      raw << '\t'
          << (kpc_read_success ? kpc_after[i] - kpc_before[i] : 0);
    }
    raw << '\n';
  }
  raw.flush();
  if (!raw) {
    std::cerr << "raw sample write failed\n";
    return 66;
  }

  mach_timebase_info_data_t timebase {};
  const kern_return_t timebase_rc = mach_timebase_info(&timebase);

  std::cout << "PIREUS_APPLE_CPU_INTERFACE_MATERIAL_PARITY_V1\n";
  std::cout << "producer_language=C++\n";
  std::cout << "producer_role=MATERIAL_PARITY\n";
  std::cout << "sounio_source_sha256=" << kSounioSourceSha256 << '\n';
  std::cout << "sounio_semantics_sha256=" << kSounioSemanticsSha256
            << '\n';
  std::cout << "authority_commit=" << kAuthorityCommit << '\n';
  std::cout << "semantic_write=false\n";
  std::cout << "expected_result_write=false\n";
  std::cout << "classification_requested=false\n";
  std::cout << "semantic_verdict_emitted=false\n";
  std::cout << "cost_present=false\n";
  std::cout << "measurand_validated=false\n";
  std::cout << "requested_warmups=" << kWarmupCount << '\n';
  std::cout << "requested_samples=" << kSampleCount << '\n';
  std::cout << "candidate_count=6\n";
  std::cout << "hostname_rc=" << hostname_rc << '\n';
  std::cout << "hostname=" << (hostname_rc == 0 ? hostname : "UNAVAILABLE")
            << '\n';
  std::cout << "uname_rc=" << uname_rc << '\n';
  std::cout << "os=" << (uname_rc == 0 ? system_name.sysname : "UNAVAILABLE")
            << '\n';
  std::cout << "os_release="
            << (uname_rc == 0 ? system_name.release : "UNAVAILABLE") << '\n';
  std::cout << "architecture="
            << (uname_rc == 0 ? system_name.machine : "UNAVAILABLE") << '\n';
  std::cout << "model=" << sysctl_string("hw.model") << '\n';
  std::cout << "cpu=" << sysctl_string("machdep.cpu.brand_string") << '\n';
  std::cout << "target=" << sysctl_string("hw.targettype") << '\n';
  std::cout << "perf_levels_available="
            << (perf_levels_available ? "true" : "false") << '\n';
  std::cout << "perf_levels=" << perf_levels << '\n';
  std::cout << "perf_levels_error=" << perf_levels_error << '\n';
  std::cout << "thread_id_rc=" << thread_id_rc << '\n';
  std::cout << "thread_id_nonzero=" << (thread_id != 0 ? "true" : "false")
            << '\n';

  std::cout << "candidate_0_family=CORE_CYCLE_COUNTER\n";
  std::cout << "candidate_0_interface=kpc_get_curcpu_counters_fixed\n";
  std::cout << "candidate_0_access_path=" << kpc.image << '\n';
  emit_bool("candidate_0_image_loaded", kpc.handle != nullptr);
  emit_bool("candidate_0_symbols_complete", kpc_symbols_complete);
  std::cout << "candidate_0_counter_count=" << kpc_counter_count << '\n';
  emit_bool("candidate_0_counter_count_valid", kpc_count_valid);
  std::cout << "candidate_0_config_count=" << kpc_config_count << '\n';
  emit_bool("candidate_0_config_count_valid", kpc_config_count_valid);
  std::cout << "candidate_0_pmu_version=" << kpc_pmu_version << '\n';
  std::cout << "candidate_0_classes_mask=" << kpc_classes << '\n';
  std::cout << "candidate_0_running_mask=" << kpc_running << '\n';
  std::cout << "candidate_0_force_all_counters=" << kpc_force_all << '\n';
  std::cout << "candidate_0_fixed_config_rc=" << kpc_fixed_config_rc << '\n';
  for (std::size_t i = 0; i < kpc_fixed_config.size(); ++i) {
    std::cout << "candidate_0_fixed_config_" << i << '='
              << kpc_fixed_config[i] << '\n';
  }
  std::cout << "candidate_0_first_before_rc=" << kpc_first_before_rc << '\n';
  std::cout << "candidate_0_first_after_rc=" << kpc_first_after_rc << '\n';
  std::cout << "candidate_0_first_before_cpu=" << kpc_first_before_cpu
            << '\n';
  std::cout << "candidate_0_first_after_cpu=" << kpc_first_after_cpu << '\n';
  std::cout << "candidate_0_read_successes=" << kpc_read_successes << '\n';
  std::cout << "candidate_0_accepted_samples=" << kpc_accepted_samples
            << '\n';
  std::cout << "candidate_0_migrations_detected_rejected=" << kpc_migrations
            << '\n';
  std::cout << "candidate_0_read_boundary="
               "atomic_signal_fence+opaque_kpc_call\n";
  std::cout << "candidate_0_core_identity_source=curcpu_out_parameter\n";
  std::cout << "candidate_0_counter_width_bits=64_api_storage\n";
  std::cout << "candidate_0_wrap_rule=unsigned_modulo_64_observation\n";
  for (std::size_t i = 0; i < kpc_deltas.size(); ++i) {
    std::cout << "candidate_0_counter_" << i
              << "_median_delta=" << median(kpc_deltas[i]) << '\n';
  }

  std::cout << "candidate_1_family=PROCESS_PMU_CYCLE_EVENT\n";
  std::cout << "candidate_1_interface=PROC_PIDTHREADCOUNTS\n";
  std::cout << "candidate_1_counter_domain=THREAD_PERF_LEVEL_AGGREGATE\n";
  std::cout << "candidate_1_first_before_rc=" << thread_first_before_rc
            << '\n';
  std::cout << "candidate_1_first_after_rc=" << thread_first_after_rc << '\n';
  std::cout << "candidate_1_first_before_errno=" << thread_first_before_error
            << '\n';
  std::cout << "candidate_1_first_after_errno=" << thread_first_after_error
            << '\n';
  std::cout << "candidate_1_returned_perf_levels="
            << thread_first_returned_levels << '\n';
  std::cout << "candidate_1_read_successes=" << thread_read_successes << '\n';
  std::cout << "candidate_1_median_cycles_delta="
            << median(thread_cycle_deltas) << '\n';
  std::cout << "candidate_1_median_instructions_delta="
            << median(thread_instruction_deltas) << '\n';
  std::cout << "candidate_1_migration_observable=false\n";

  const bool xctrace_system = access("/usr/bin/xctrace", X_OK) == 0;
  const bool xctrace_xcode =
      access("/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace",
             X_OK) == 0;
  std::cout << "candidate_2_family=SYSTEM_TRACE_CYCLE_EVENT\n";
  std::cout << "candidate_2_interface=xctrace\n";
  emit_bool("candidate_2_system_path_executable", xctrace_system);
  emit_bool("candidate_2_xcode_path_executable", xctrace_xcode);
  std::cout << "candidate_2_trace_executed=false\n";
  std::cout << "candidate_2_event_configuration=ABSENT\n";

  std::cout << "candidate_3_family=ARCHITECTURAL_TIMER_TICK\n";
  std::cout << "candidate_3_interface=mach_absolute_time\n";
  std::cout << "candidate_3_native_unit=TIMER_TICK\n";
  std::cout << "candidate_3_timebase_rc=" << timebase_rc << '\n';
  std::cout << "candidate_3_timebase_numer=" << timebase.numer << '\n';
  std::cout << "candidate_3_timebase_denom=" << timebase.denom << '\n';
  std::cout << "candidate_3_median_tick_delta="
            << median(absolute_time_deltas) << '\n';

  std::cout << "candidate_4_family=OS_MONOTONIC_TIME\n";
  std::cout << "candidate_4_interface=clock_gettime_nsec_np_CLOCK_UPTIME_RAW\n";
  std::cout << "candidate_4_native_unit=NANOSECOND\n";
  std::cout << "candidate_4_read_successes=" << uptime_raw_successes << '\n';
  std::cout << "candidate_4_median_nanosecond_delta="
            << median(uptime_raw_deltas) << '\n';

  std::cout << "candidate_5_family=FREQUENCY_DERIVED_ESTIMATE\n";
  std::cout << "candidate_5_interface=sysctl_hw_cpufrequency\n";
  emit_bool("candidate_5_frequency_available", cpu_frequency_available);
  std::cout << "candidate_5_frequency_errno=" << cpu_frequency_error << '\n';
  std::cout << "candidate_5_frequency_hz=" << cpu_frequency << '\n';
  std::cout << "candidate_5_native_cycle_claim=false\n";

  std::cout << "raw_sample_path=" << argv[1] << '\n';
  std::cout << "dependency_sink=" << sink << '\n';
  std::cout << "probe_completed=true\n";

  if (kpc.handle != nullptr) {
    dlclose(kpc.handle);
  }
  return 0;
}
