// C++ MATERIAL_PARITY probe for frozen Sounio U250 launch semantics.
// It emits XRT lifecycle facts and opaque output values only.

#include "material_sha256.hpp"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr std::uintmax_t kArtifactSize = 41112056;
constexpr char kArtifactDigest[] =
    "d30078c7b2e8690aef892b4b6cf96af0f490b70e2b367e5e3679be04fcd4bdbf";
constexpr char kXclbinUuid[] = "c50267ec-ae68-48a1-1559-3473f046689c";
constexpr char kDeviceBdf[] = "0000:d8:00.1";
constexpr char kManagementBdf[] = "0000:d8:00.0";
constexpr char kCardSerial[] = "22000321B01F";
constexpr char kKernel[] = "krnl_san_scan";
constexpr char kParentSemantics[] =
    "8d89d94c4b808548a9b8827ac6f54c29d510a40674f2421e99b497f8a6f32f05";
constexpr char kParentFreeze[] =
    "662fe6e6f2dc0c5b27f227c26e5d92d1f0da3c588e7ef94accbf4b49b8abd2e5";
constexpr char kToolchainDigest[] =
    "683759ae2d3f34e05a780a935399bb3b12c3acbf3d6a81d863620b572d231e3e";
constexpr char kHardwareDigest[] =
    "02bde35408178d3a691d69a8f4f10099c45ab23f468d673cb27f4fedc798554a";
constexpr char kCommandDigest[] =
    "5bb07529f0f6bb2536395957034f8703854cc7e1e5e3f833afecf9523c964579";
constexpr char kResultContractDigest[] =
    "7bd0d2d7b7838e046a619960c149622b95b5ab4856eb496ff398daaa48d70594";
constexpr std::uint32_t kNSamples = 4;
constexpr std::uint32_t kNPoints = 2;
constexpr std::uint32_t kQDelta = 16384;
constexpr unsigned int kRunTimeoutMs = 5000;

struct ToolResult {
  int exit_code = 127;
  std::string output;
};

struct LaunchFacts {
  bool device_programmed = false;
  bool kernel_opened = false;
  int buffers_allocated = 0;
  bool inputs_synced = false;
  bool run_submitted = false;
  bool run_completed = false;
  int outputs_synced = 0;
  bool output_values_recorded = false;
  std::array<std::uint32_t, 8> histogram{};
  std::uint32_t catastrophes = 0;
  std::uint64_t flops = 0;
  std::string loaded_uuid = "unavailable";
};

std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  pireus::material::Sha256 sha;
  std::array<std::uint8_t, 1U << 16U> buffer{};
  while (input) {
    input.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    const std::streamsize count = input.gcount();
    if (count > 0) {
      sha.update(buffer.data(), static_cast<std::size_t>(count));
    }
  }
  return pireus::material::digest_hex(sha.finish());
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    return {};
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

ToolResult run_tool(const char* tool, const std::vector<std::string>& args) {
  int output_pipe[2] = {-1, -1};
  if (pipe(output_pipe) != 0) {
    return {};
  }
  const pid_t child = fork();
  if (child < 0) {
    close(output_pipe[0]);
    close(output_pipe[1]);
    return {};
  }
  if (child == 0) {
    (void)dup2(output_pipe[1], STDOUT_FILENO);
    (void)dup2(output_pipe[1], STDERR_FILENO);
    close(output_pipe[0]);
    close(output_pipe[1]);
    std::vector<char*> argv;
    argv.reserve(args.size() + 2);
    argv.push_back(const_cast<char*>(tool));
    for (const std::string& arg : args) {
      argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);
    execv(tool, argv.data());
    _exit(127);
  }
  close(output_pipe[1]);
  ToolResult result;
  std::array<char, 8192> buffer{};
  while (true) {
    const ssize_t count = read(output_pipe[0], buffer.data(), buffer.size());
    if (count > 0) {
      result.output.append(buffer.data(), static_cast<std::size_t>(count));
      continue;
    }
    if (count < 0 && errno == EINTR) {
      continue;
    }
    break;
  }
  close(output_pipe[0]);
  int status = 0;
  while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
  result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128;
  return result;
}

bool contains(const std::string& text, std::string_view needle) {
  return text.find(needle) != std::string::npos;
}

void emit_bool(const char* key, bool value) {
  std::cout << key << '=' << (value ? "true" : "false") << '\n';
}

void emit_histogram(const std::array<std::uint32_t, 8>& histogram,
                    bool available) {
  std::cout << "histogram_observed=";
  if (!available) {
    std::cout << "unavailable\n";
    return;
  }
  for (std::size_t i = 0; i < histogram.size(); ++i) {
    if (i != 0) {
      std::cout << ',';
    }
    std::cout << histogram[i];
  }
  std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: u250_kernel_launch_probe XCLBIN\n";
    return 64;
  }

  const std::filesystem::path artifact = argv[1];
  std::error_code size_error;
  const std::uintmax_t artifact_size =
      std::filesystem::file_size(artifact, size_error);
  const std::string artifact_digest = sha256_file(artifact);
  const std::string xrt_version = read_text("/opt/xilinx/xrt/version.json");
  const ToolResult management = run_tool(
      "/opt/xilinx/xrt/bin/unwrapped/loader",
      {"-exec", "xbmgmt", "examine", "-d", kManagementBdf,
       "--report", "platform"});
  std::array<char, 256> hostname{};
  const bool hostname_valid =
      gethostname(hostname.data(), hostname.size() - 1) == 0 &&
      std::string(hostname.data()) == "dl380-proxmox";
  const bool preflight_valid = !size_error && artifact_size == kArtifactSize &&
      artifact_digest == kArtifactDigest && hostname_valid &&
      contains(xrt_version, "\"BUILD_VERSION\" : \"2.23.0\"") &&
      management.exit_code == 0 &&
      contains(management.output, "Serial Number        : 22000321B01F");

  LaunchFacts facts;
  if (preflight_valid) {
    try {
      auto device = xrt::device(kDeviceBdf);
      const auto uuid = device.load_xclbin(artifact.string());
      facts.loaded_uuid = uuid.to_string();
      facts.device_programmed = facts.loaded_uuid == kXclbinUuid;
      if (facts.device_programmed) {
        auto kernel = xrt::kernel(device, uuid, kKernel);
        facts.kernel_opened = true;

        std::array<std::uint64_t, 8> samples{};
        std::array<std::uint64_t, 8> lut = {8192, 16384, 0, 0, 0, 0, 0, 0};
        auto bo_samples = xrt::bo(
            device, samples.size() * sizeof(samples[0]), kernel.group_id(0));
        auto bo_lut = xrt::bo(
            device, lut.size() * sizeof(lut[0]), kernel.group_id(1));
        auto bo_hist = xrt::bo(
            device, facts.histogram.size() * sizeof(facts.histogram[0]),
            kernel.group_id(5));
        auto bo_cat = xrt::bo(
            device, sizeof(facts.catastrophes), kernel.group_id(6));
        auto bo_flops = xrt::bo(
            device, sizeof(facts.flops), kernel.group_id(7));
        facts.buffers_allocated = 5;

        std::memcpy(bo_samples.map<void*>(), samples.data(),
                    samples.size() * sizeof(samples[0]));
        std::memcpy(bo_lut.map<void*>(), lut.data(),
                    lut.size() * sizeof(lut[0]));
        bo_samples.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_lut.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        facts.inputs_synced = true;

        auto run = kernel(bo_samples, bo_lut, kQDelta, kNPoints, kNSamples,
                          bo_hist, bo_cat, bo_flops);
        facts.run_submitted = true;
        const ert_cmd_state state = run.wait(kRunTimeoutMs);
        facts.run_completed = state == ERT_CMD_STATE_COMPLETED;
        if (!facts.run_completed) {
          (void)run.abort();
        } else {
          bo_hist.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
          ++facts.outputs_synced;
          bo_cat.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
          ++facts.outputs_synced;
          bo_flops.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
          ++facts.outputs_synced;
          std::memcpy(facts.histogram.data(), bo_hist.map<void*>(),
                      facts.histogram.size() * sizeof(facts.histogram[0]));
          std::memcpy(&facts.catastrophes, bo_cat.map<void*>(),
                      sizeof(facts.catastrophes));
          std::memcpy(&facts.flops, bo_flops.map<void*>(), sizeof(facts.flops));
          facts.output_values_recorded = true;
        }
      }
    } catch (const std::exception& error) {
      std::cerr << "XRT launch exception: " << error.what() << '\n';
    }
  }

  const bool probe_valid = preflight_valid && facts.device_programmed &&
      facts.kernel_opened && facts.buffers_allocated == 5 &&
      facts.inputs_synced && facts.run_submitted && facts.run_completed &&
      facts.outputs_synced == 3 && facts.output_values_recorded;

  std::cout << "schema=pireus.u250.kernel-launch-probe.v0\n";
  std::cout << "producer_language=C++\n";
  std::cout << "producer_role=MATERIAL_PARITY\n";
  std::cout << "semantic_authority_language=Sounio\n";
  std::cout << "parent_semantics_sha256=" << kParentSemantics << '\n';
  std::cout << "parent_freeze_sha256=" << kParentFreeze << '\n';
  std::cout << "artifact_name=krnl_san_scan.hw.xclbin\n";
  std::cout << "artifact_sha256=" << artifact_digest << '\n';
  std::cout << "xclbin_uuid=" << facts.loaded_uuid << '\n';
  std::cout << "device_bdf=" << kDeviceBdf << '\n';
  std::cout << "card_serial=" << kCardSerial << '\n';
  std::cout << "xrt_version=2.23.0\n";
  std::cout << "kernel=" << kKernel << '\n';
  std::cout << "n_samples=" << kNSamples << '\n';
  std::cout << "n_points=" << kNPoints << '\n';
  std::cout << "q_delta=" << kQDelta << '\n';
  std::cout << "packed_sample_beats=1\n";
  std::cout << "sample_bytes=64\n";
  std::cout << "samples_all_zero=true\n";
  std::cout << "lut=8192,16384,0,0,0,0,0,0\n";
  std::cout << "toolchain_sha256=" << kToolchainDigest << '\n';
  std::cout << "hardware_sha256=" << kHardwareDigest << '\n';
  std::cout << "command_sha256=" << kCommandDigest << '\n';
  std::cout << "result_contract_sha256=" << kResultContractDigest << '\n';
  emit_bool("device_programmed", facts.device_programmed);
  emit_bool("kernel_opened", facts.kernel_opened);
  std::cout << "buffers_allocated=" << facts.buffers_allocated << '\n';
  emit_bool("inputs_synced", facts.inputs_synced);
  emit_bool("run_submitted", facts.run_submitted);
  emit_bool("run_completed", facts.run_completed);
  std::cout << "outputs_synced=" << facts.outputs_synced << '\n';
  std::cout << "histogram_output_bytes=32\n";
  std::cout << "catastrophe_output_bytes=4\n";
  std::cout << "flops_output_bytes=8\n";
  emit_histogram(facts.histogram, facts.output_values_recorded);
  std::cout << "catastrophe_observed=";
  if (facts.output_values_recorded) {
    std::cout << facts.catastrophes << '\n';
  } else {
    std::cout << "unavailable\n";
  }
  std::cout << "flops_observed=";
  if (facts.output_values_recorded) {
    std::cout << facts.flops << '\n';
  } else {
    std::cout << "unavailable\n";
  }
  emit_bool("output_values_recorded", facts.output_values_recorded);
  emit_bool("probe_valid", probe_valid);
  emit_bool("semantic_verdict_emitted", false);
  emit_bool("expected_output_present", false);
  emit_bool("kernel_correctness_present", false);
  std::cout << "operation_capability_count=0\n";
  emit_bool("lowering_requested", false);
  emit_bool("isa_claim_present", false);
  emit_bool("performance_present", false);
  emit_bool("claim_ready", false);
  return probe_valid ? 0 : 1;
}
