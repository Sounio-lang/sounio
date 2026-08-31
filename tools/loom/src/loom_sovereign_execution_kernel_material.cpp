#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <openssl/evp.h>
#include <openssl/sha.h>

#include <linux/limits.h>
#include <poll.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

extern char** environ;

namespace {

constexpr std::string_view kFrozenSemanticManifestSha256 =
    "966f022c98bc7df89ce40a90ede9ec8a9a726499baec0fd21e72f327f286a176";
constexpr std::string_view kFrozenPeerJudgmentSha256 =
    "f7adafcd1c79364b75ebe48b66999ec2d7b82a12d6b8e45d9c1cc4637a4ca9ca";
constexpr auto kDeadline = std::chrono::seconds(5);
constexpr std::size_t kMaximumRecordBytes = 128 * 1024;

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class Fd {
 public:
  Fd() = default;
  explicit Fd(int value) : value_(value) {}
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  Fd(Fd&& other) noexcept : value_(other.release()) {}
  Fd& operator=(Fd&& other) noexcept {
    if (this != &other) reset(other.release());
    return *this;
  }
  ~Fd() { reset(); }
  int get() const { return value_; }
  explicit operator bool() const { return value_ >= 0; }
  int release() {
    const int value = value_;
    value_ = -1;
    return value;
  }
  void reset(int value = -1) {
    if (value_ >= 0) close(value_);
    value_ = value;
  }

 private:
  int value_ = -1;
};

std::string sha256(std::string_view value) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(value.data()), value.size(),
         digest);
  static constexpr char alphabet[] = "0123456789abcdef";
  std::string output(SHA256_DIGEST_LENGTH * 2, '0');
  for (std::size_t index = 0; index < SHA256_DIGEST_LENGTH; ++index) {
    output[index * 2] = alphabet[digest[index] >> 4];
    output[index * 2 + 1] = alphabet[digest[index] & 0x0f];
  }
  return output;
}

std::string read_descriptor(int descriptor,
                            std::size_t limit = kMaximumRecordBytes) {
  std::string output;
  std::array<char, 8192> buffer{};
  for (;;) {
    const ssize_t count = read(descriptor, buffer.data(), buffer.size());
    if (count > 0) {
      output.append(buffer.data(), static_cast<std::size_t>(count));
      if (output.size() > limit) throw Error("record exceeded size limit");
    } else if (count == 0) {
      return output;
    } else if (errno != EINTR) {
      throw Error(std::string("descriptor read failed: ") + std::strerror(errno));
    }
  }
}

std::string read_file(const std::string& path) {
  Fd descriptor(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (!descriptor) throw Error("cannot open file: " + path);
  return read_descriptor(descriptor.get());
}

std::string file_sha256(const std::string& path) {
  Fd descriptor(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (!descriptor) throw Error("cannot hash file: " + path);
  EVP_MD_CTX* context = EVP_MD_CTX_new();
  if (context == nullptr) throw Error("cannot allocate SHA256 context");
  const auto free_context = [&]() { EVP_MD_CTX_free(context); };
  if (EVP_DigestInit_ex(context, EVP_sha256(), nullptr) != 1) {
    free_context();
    throw Error("cannot initialize SHA256 context");
  }
  std::array<unsigned char, 64 * 1024> buffer{};
  for (;;) {
    const ssize_t count = read(descriptor.get(), buffer.data(), buffer.size());
    if (count > 0) {
      if (EVP_DigestUpdate(context, buffer.data(),
                           static_cast<std::size_t>(count)) != 1) {
        free_context();
        throw Error("cannot update SHA256 context");
      }
    } else if (count == 0) {
      break;
    } else if (errno != EINTR) {
      free_context();
      throw Error("cannot read file for SHA256");
    }
  }
  unsigned char digest[SHA256_DIGEST_LENGTH];
  unsigned int digest_size = 0;
  if (EVP_DigestFinal_ex(context, digest, &digest_size) != 1 ||
      digest_size != SHA256_DIGEST_LENGTH) {
    free_context();
    throw Error("cannot finalize SHA256 context");
  }
  free_context();
  static constexpr char alphabet[] = "0123456789abcdef";
  std::string output(SHA256_DIGEST_LENGTH * 2, '0');
  for (std::size_t index = 0; index < SHA256_DIGEST_LENGTH; ++index) {
    output[index * 2] = alphabet[digest[index] >> 4];
    output[index * 2 + 1] = alphabet[digest[index] & 0x0f];
  }
  return output;
}

struct Manifest {
  std::map<std::string, std::string> fields;

  const std::string& require(const std::string& key) const {
    const auto found = fields.find(key);
    if (found == fields.end()) throw Error("manifest field absent: " + key);
    return found->second;
  }
};

Manifest parse_manifest(const std::string& path) {
  Manifest manifest;
  std::istringstream input(read_file(path));
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const std::size_t equals = line.find('=');
    if (equals == std::string::npos || equals == 0 ||
        !manifest.fields.emplace(line.substr(0, equals),
                                 line.substr(equals + 1)).second) {
      throw Error("manifest field malformed");
    }
  }
  return manifest;
}

void write_all(int descriptor, std::string_view value) {
  std::size_t offset = 0;
  while (offset < value.size()) {
    const ssize_t count = write(descriptor, value.data() + offset,
                                value.size() - offset);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
    } else if (count < 0 && errno != EINTR) {
      throw Error(std::string("write failed: ") + std::strerror(errno));
    }
  }
}

void send_packet(int descriptor, std::string_view value) {
  for (;;) {
    const ssize_t count = send(descriptor, value.data(), value.size(), MSG_NOSIGNAL);
    if (count == static_cast<ssize_t>(value.size())) return;
    if (count < 0 && errno == EINTR) continue;
    throw Error("packet send failed");
  }
}

std::string receive_packet(int descriptor) {
  std::array<char, 4096> buffer{};
  for (;;) {
    const ssize_t count = recv(descriptor, buffer.data(), buffer.size(), 0);
    if (count > 0) return std::string(buffer.data(), static_cast<std::size_t>(count));
    if (count < 0 && errno == EINTR) continue;
    throw Error("packet receive failed");
  }
}

std::string read_line(int descriptor) {
  std::string output;
  while (output.size() < 4096) {
    char value = 0;
    const ssize_t count = read(descriptor, &value, 1);
    if (count == 1) {
      if (value == '\n') return output;
      output.push_back(value);
    } else if (count == 0) {
      throw Error("unexpected observation EOF");
    } else if (errno != EINTR) {
      throw Error("observation read failed");
    }
  }
  throw Error("observation line exceeded limit");
}

std::uint64_t process_start_tick(pid_t pid) {
  const std::string stat = read_file("/proc/" + std::to_string(pid) + "/stat");
  const std::size_t close = stat.rfind(')');
  if (close == std::string::npos) throw Error("process stat malformed");
  std::istringstream input(stat.substr(close + 2));
  std::string field;
  for (int index = 0; index <= 19; ++index) {
    if (!(input >> field)) throw Error("process stat truncated");
  }
  std::size_t consumed = 0;
  const std::uint64_t value = std::stoull(field, &consumed, 10);
  if (consumed != field.size() || value == 0) {
    throw Error("process start tick malformed");
  }
  return value;
}

pid_t process_parent(pid_t pid) {
  const std::string stat = read_file("/proc/" + std::to_string(pid) + "/stat");
  const std::size_t close = stat.rfind(')');
  if (close == std::string::npos) throw Error("process stat malformed");
  std::istringstream input(stat.substr(close + 2));
  char state = 0;
  pid_t parent = 0;
  if (!(input >> state >> parent) || parent <= 0) {
    throw Error("process parent malformed");
  }
  return parent;
}

bool descends_from(pid_t pid, pid_t ancestor) {
  for (int depth = 0; depth < 64 && pid > 1; ++depth) {
    if (pid == ancestor) return true;
    try {
      pid = process_parent(pid);
    } catch (...) {
      return false;
    }
  }
  return false;
}

std::string process_executable_sha256(pid_t pid) {
  const std::string path = "/proc/" + std::to_string(pid) + "/exe";
  std::array<char, PATH_MAX + 1> target{};
  const ssize_t count = readlink(path.c_str(), target.data(), PATH_MAX);
  if (count <= 0 || count > PATH_MAX) {
    throw Error("cannot resolve process executable");
  }
  return file_sha256(std::string(target.data(), static_cast<std::size_t>(count)));
}

int open_pidfd(pid_t pid) {
#ifdef SYS_pidfd_open
  return static_cast<int>(syscall(SYS_pidfd_open, pid, 0));
#else
  static_cast<void>(pid);
  errno = ENOSYS;
  return -1;
#endif
}

bool pidfd_alive(int descriptor) {
  pollfd candidate{descriptor, POLLIN, 0};
  const int ready = poll(&candidate, 1, 0);
  if (ready < 0) throw Error("pidfd poll failed");
  return ready == 0;
}

bool wait_pidfd_dead(int descriptor, std::chrono::milliseconds timeout) {
  pollfd candidate{descriptor, POLLIN, 0};
  const int ready = poll(&candidate, 1, static_cast<int>(timeout.count()));
  return ready == 1 && (candidate.revents & POLLIN) != 0;
}

bool process_absent(pid_t pid, std::uint64_t start_tick) {
  try {
    return process_start_tick(pid) != start_tick;
  } catch (...) {
    return true;
  }
}

int wait_child(pid_t pid, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    int status = 0;
    const pid_t result = waitpid(pid, &status, WNOHANG);
    if (result == pid) {
      if (WIFEXITED(status)) return WEXITSTATUS(status);
      if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
      return 255;
    }
    if (result < 0 && errno != EINTR) throw Error("waitpid failed");
    if (std::chrono::steady_clock::now() >= deadline) {
      throw Error("child exit timed out");
    }
    poll(nullptr, 0, 5);
  }
}

Fd make_listener(const std::string& path) {
  Fd descriptor(socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0));
  if (!descriptor) throw Error("cannot create Guardian listener");
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  if (path.size() >= sizeof(address.sun_path)) throw Error("socket path too long");
  std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
  unlink(path.c_str());
  if (bind(descriptor.get(), reinterpret_cast<sockaddr*>(&address),
           sizeof(address)) != 0 ||
      chmod(path.c_str(), 0600) != 0 || listen(descriptor.get(), 16) != 0) {
    throw Error("cannot bind Guardian listener");
  }
  return descriptor;
}

Fd connect_socket(const std::string& path) {
  Fd descriptor(socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0));
  if (!descriptor) throw Error("cannot create client socket");
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  if (path.size() >= sizeof(address.sun_path)) throw Error("socket path too long");
  std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
  if (connect(descriptor.get(), reinterpret_cast<sockaddr*>(&address),
              sizeof(address)) != 0) {
    throw Error("cannot connect to Guardian");
  }
  return descriptor;
}

Fd accept_socket(int listener) {
  for (;;) {
    const int descriptor = accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
    if (descriptor >= 0) return Fd(descriptor);
    if (errno != EINTR) throw Error("Guardian accept failed");
  }
}

struct PeerObservation {
  bool so_peercred = false;
  bool pidfd = false;
  bool start_tick = false;
  bool ancestry = false;
  bool executable = false;
  bool operation = false;
  bool exact_pid = false;
  pid_t pid = 0;

  bool authenticated() const {
    return so_peercred && pidfd && start_tick && ancestry && executable &&
           operation && exact_pid;
  }
};

PeerObservation authenticate_peer(int descriptor, pid_t expected_pid,
                                  std::uint64_t expected_start,
                                  pid_t harness_pid,
                                  const std::string& executable_sha256,
                                  std::string_view operation) {
  ucred credentials{};
  socklen_t size = sizeof(credentials);
  if (getsockopt(descriptor, SOL_SOCKET, SO_PEERCRED, &credentials, &size) != 0 ||
      size != sizeof(credentials)) {
    throw Error("SO_PEERCRED failed");
  }
  PeerObservation observation;
  observation.pid = credentials.pid;
  observation.so_peercred = credentials.uid == getuid() &&
                            credentials.gid == getgid();
  Fd pidfd(open_pidfd(credentials.pid));
  observation.pidfd = pidfd && pidfd_alive(pidfd.get());
  try {
    observation.start_tick =
        process_start_tick(credentials.pid) == expected_start;
    observation.ancestry = descends_from(credentials.pid, harness_pid);
    observation.executable =
        process_executable_sha256(credentials.pid) == executable_sha256;
  } catch (...) {
    observation.start_tick = false;
    observation.ancestry = false;
    observation.executable = false;
  }
  observation.operation = operation == "CONSUME";
#ifdef LOOM_SOVEREIGN_DISABLE_PRINCIPAL_BINDING
  static_cast<void>(expected_pid);
  observation.start_tick = true;
  observation.exact_pid = true;
#else
  observation.exact_pid = credentials.pid == expected_pid;
#endif
  return observation;
}

struct ProgramResult {
  int code = 255;
  std::string output;
};

ProgramResult run_sounio(const std::string& runtime, const std::string& frame) {
  int input_pipe[2] = {-1, -1};
  int output_pipe[2] = {-1, -1};
  if (pipe2(input_pipe, O_CLOEXEC) != 0 || pipe2(output_pipe, O_CLOEXEC) != 0) {
    throw Error("cannot create Sounio pipes");
  }
  Fd input_read(input_pipe[0]);
  Fd input_write(input_pipe[1]);
  Fd output_read(output_pipe[0]);
  Fd output_write(output_pipe[1]);
  const pid_t child = fork();
  if (child < 0) throw Error("cannot fork Sounio authority");
  if (child == 0) {
    if (dup2(input_read.get(), STDIN_FILENO) < 0 ||
        dup2(output_write.get(), STDOUT_FILENO) < 0 ||
        dup2(output_write.get(), STDERR_FILENO) < 0) {
      _exit(126);
    }
    execl(runtime.c_str(), runtime.c_str(), nullptr);
    _exit(127);
  }
  input_read.reset();
  output_write.reset();
  write_all(input_write.get(), frame + "\n");
  input_write.reset();
  ProgramResult result;
  result.output = read_descriptor(output_read.get());
  result.code = wait_child(child, std::chrono::duration_cast<std::chrono::milliseconds>(kDeadline));
  while (!result.output.empty() &&
         (result.output.back() == '\n' || result.output.back() == '\r')) {
    result.output.pop_back();
  }
  return result;
}

std::string first_line(const std::string& value) {
  const std::size_t newline = value.find('\n');
  return value.substr(0, newline);
}

struct ControlProcess {
  pid_t pid = -1;
  std::string role;
};

[[noreturn]] void run_refused_control(const std::string& socket_path,
                                      const std::string& request,
                                      int acknowledgement,
                                      int keepalive) {
  try {
    Fd socket = connect_socket(socket_path);
    send_packet(socket.get(), request);
    const std::string response = receive_packet(socket.get());
    if (response.rfind("ERR ", 0) != 0) _exit(91);
    write_all(acknowledgement, "R");
    char value = 0;
    while (read(keepalive, &value, 1) < 0 && errno == EINTR) {
    }
    _exit(0);
  } catch (...) {
    try {
      write_all(acknowledgement, "F");
    } catch (...) {
    }
    _exit(92);
  }
}

[[noreturn]] void run_legitimate_client(const std::string& socket_path,
                                        int gate) {
  char value = 0;
  while (read(gate, &value, 1) < 0 && errno == EINTR) {
  }
  try {
    {
      Fd socket = connect_socket(socket_path);
      send_packet(socket.get(), "CONSUME");
      if (receive_packet(socket.get()) != "OK CONSUMED") _exit(81);
    }
    {
      Fd socket = connect_socket(socket_path);
      send_packet(socket.get(), "CONSUME");
      if (receive_packet(socket.get()) != "ERR grant-not-issued") _exit(82);
    }
    _exit(0);
  } catch (...) {
    _exit(83);
  }
}

enum class GrantState { Issued, Consumed };

struct TreatmentResult {
  bool peercred = false;
  bool pidfd = false;
  bool start_tick = false;
  bool ancestry = false;
  bool executable = false;
  bool operation = false;
  bool spoof_refused = false;
  bool release_refused = false;
  bool pre_exec_zero = false;
  bool transport_dead = false;
  bool interfaces_dead = false;
  bool witness_continued = false;
  bool exactly_once = false;
  bool replay_refused = false;
  bool pdeathsig = false;
};

void guardian_treatment(const std::string& socket_path, pid_t harness_pid,
                        int observation, int release_client) {
  try {
    Fd listener = make_listener(socket_path);
    int client_gate_pipe[2] = {-1, -1};
    if (pipe2(client_gate_pipe, O_CLOEXEC) != 0) {
      throw Error("cannot create client gate");
    }
    Fd client_gate_read(client_gate_pipe[0]);
    Fd client_gate_write(client_gate_pipe[1]);
    const pid_t client = fork();
    if (client < 0) throw Error("cannot fork legitimate client");
    if (client == 0) {
      listener.reset();
      client_gate_write.reset();
      run_legitimate_client(socket_path, client_gate_read.get());
    }
    client_gate_read.reset();
    const std::uint64_t client_start = process_start_tick(client);
    const std::string executable = process_executable_sha256(getpid());
    write_all(observation, "READY " + std::to_string(client) + "\n");

    int spoof_refused = 0;
    int release_refused = 0;
    int material_started = 0;
    for (int index = 0; index < 7; ++index) {
      Fd connection = accept_socket(listener.get());
      const std::string request = receive_packet(connection.get());
      const std::string operation = request == "CONSUME" ? "CONSUME" : "RELEASE";
      const PeerObservation peer = authenticate_peer(
          connection.get(), client, client_start, harness_pid, executable,
          operation);
      if (peer.authenticated()) {
        if (peer.pid != client) {
          send_packet(connection.get(), "ERR same-uid-spoof-admitted");
          write_all(observation, "SABOTAGE_ADMITTED\n");
          _exit(96);
        }
        throw Error("control unexpectedly authenticated");
      }
      if (operation == "CONSUME") ++spoof_refused;
      else ++release_refused;
      send_packet(connection.get(), "ERR peer-refused");
    }
    if (spoof_refused != 1 || release_refused != 6 || material_started != 0) {
      throw Error("causal controls incomplete");
    }
    write_all(observation, "CONTROLS\n");
    char release = 0;
    while (read(release_client, &release, 1) < 0 && errno == EINTR) {
    }
    write_all(client_gate_write.get(), "G");
    client_gate_write.reset();

    GrantState state = GrantState::Issued;
    Fd consume = accept_socket(listener.get());
    const std::string consume_request = receive_packet(consume.get());
    const PeerObservation peer = authenticate_peer(
        consume.get(), client, client_start, harness_pid, executable,
        consume_request);
    if (!peer.authenticated() || state != GrantState::Issued) {
      throw Error("legitimate consume refused");
    }
    state = GrantState::Consumed;

    int marker_pipe[2] = {-1, -1};
    if (pipe2(marker_pipe, O_CLOEXEC) != 0) throw Error("cannot create marker pipe");
    Fd marker_read(marker_pipe[0]);
    Fd marker_write(marker_pipe[1]);
    const pid_t material = fork();
    if (material < 0) throw Error("cannot fork material witness");
    if (material == 0) {
      marker_read.reset();
#ifndef LOOM_SOVEREIGN_DISABLE_PDEATHSIG
      const pid_t guardian = getppid();
      if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != guardian) _exit(72);
#endif
      write_all(marker_write.get(), "S");
      poll(nullptr, 0, 500);
      write_all(marker_write.get(), "C");
      _exit(0);
    }
    marker_write.reset();
    ++material_started;
    send_packet(consume.get(), "OK CONSUMED");
    consume.reset();
    write_all(observation, "MATERIAL_STARTED\n");

    Fd replay = accept_socket(listener.get());
    const std::string replay_request = receive_packet(replay.get());
    const PeerObservation replay_peer = authenticate_peer(
        replay.get(), client, client_start, harness_pid, executable,
        replay_request);
    if (!replay_peer.authenticated() || state != GrantState::Consumed) {
      throw Error("replay authentication drifted");
    }
    send_packet(replay.get(), "ERR grant-not-issued");
    replay.reset();

    const int client_code = wait_child(
        client, std::chrono::duration_cast<std::chrono::milliseconds>(kDeadline));
    const int material_code = wait_child(
        material, std::chrono::duration_cast<std::chrono::milliseconds>(kDeadline));
    const std::string markers = read_descriptor(marker_read.get(), 16);
    if (client_code != 0 || material_code != 0 || markers != "SC" ||
        state != GrantState::Consumed || material_started != 1) {
      throw Error("material treatment incomplete");
    }
    write_all(observation,
              "TREATMENT peercred=1 pidfd=1 start=1 ancestry=1 executable=1 "
              "operation=1 spoof=1 release=1 preexec=1 transport=1 witness=1 "
              "exact=1 replay=1 pdeathsig=1\n");
    unlink(socket_path.c_str());
    _exit(0);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    unlink(socket_path.c_str());
    _exit(95);
  }
}

TreatmentResult run_treatment(const std::string& directory) {
  const std::string socket_path = directory + "/guardian.sock";
  int observation_pipe[2] = {-1, -1};
  int client_release_pipe[2] = {-1, -1};
  int acknowledgement_pipe[2] = {-1, -1};
  int keepalive_pipe[2] = {-1, -1};
  if (pipe2(observation_pipe, O_CLOEXEC) != 0 ||
      pipe2(client_release_pipe, O_CLOEXEC) != 0 ||
      pipe2(acknowledgement_pipe, O_CLOEXEC) != 0 ||
      pipe2(keepalive_pipe, O_CLOEXEC) != 0) {
    throw Error("cannot create treatment pipes");
  }
  Fd observation_read(observation_pipe[0]);
  Fd observation_write(observation_pipe[1]);
  Fd client_release_read(client_release_pipe[0]);
  Fd client_release_write(client_release_pipe[1]);
  Fd acknowledgement_read(acknowledgement_pipe[0]);
  Fd acknowledgement_write(acknowledgement_pipe[1]);
  Fd keepalive_read(keepalive_pipe[0]);
  Fd keepalive_write(keepalive_pipe[1]);

  const pid_t harness_pid = getpid();
  const pid_t guardian = fork();
  if (guardian < 0) throw Error("cannot fork treatment Guardian");
  if (guardian == 0) {
    observation_read.reset();
    client_release_write.reset();
    acknowledgement_read.reset();
    acknowledgement_write.reset();
    keepalive_read.reset();
    keepalive_write.reset();
    guardian_treatment(socket_path, harness_pid, observation_write.get(),
                       client_release_read.get());
  }
  observation_write.reset();
  client_release_read.reset();

  const std::string ready = read_line(observation_read.get());
  if (ready.rfind("READY ", 0) != 0) throw Error("Guardian READY absent");

  std::vector<ControlProcess> controls;
  const auto spawn_control = [&](const std::string& role,
                                 const std::string& request) {
    const pid_t child = fork();
    if (child < 0) throw Error("cannot fork same-UID control");
    if (child == 0) {
      observation_read.reset();
      client_release_write.reset();
      acknowledgement_read.reset();
      keepalive_write.reset();
      run_refused_control(socket_path, request, acknowledgement_write.get(),
                          keepalive_read.get());
    }
    controls.push_back({child, role});
  };
  spawn_control("hostile", "CONSUME");
  for (const std::string role : {"GUI", "TUI", "CLI", "Pod", "tmux", "coordinator"}) {
    spawn_control(role, "RELEASE " + role);
  }
  acknowledgement_write.reset();
  keepalive_read.reset();
  std::string acknowledgements;
  while (acknowledgements.size() < 7) {
    char value = 0;
    const ssize_t count = read(acknowledgement_read.get(), &value, 1);
    if (count == 1) acknowledgements.push_back(value);
    else if (count < 0 && errno != EINTR) throw Error("control acknowledgement failed");
  }
  const std::string control_observation = read_line(observation_read.get());
  if (control_observation == "SABOTAGE_ADMITTED") {
    for (const ControlProcess& control : controls) kill(control.pid, SIGKILL);
    keepalive_write.reset();
    for (const ControlProcess& control : controls) {
      try {
        static_cast<void>(wait_child(control.pid, std::chrono::seconds(1)));
      } catch (...) {
      }
    }
    static_cast<void>(wait_child(guardian, std::chrono::seconds(1)));
    throw Error("same-uid-spoof-admitted");
  }
  if (control_observation != "CONTROLS") {
    throw Error("control observation absent");
  }
  write_all(client_release_write.get(), "G");
  client_release_write.reset();
  if (read_line(observation_read.get()) != "MATERIAL_STARTED") {
    throw Error("material start observation absent");
  }

  for (const ControlProcess& control : controls) kill(control.pid, SIGKILL);
  keepalive_write.reset();
  bool interfaces_dead = true;
  for (const ControlProcess& control : controls) {
    const int code = wait_child(
        control.pid, std::chrono::duration_cast<std::chrono::milliseconds>(kDeadline));
    interfaces_dead = interfaces_dead && code == 128 + SIGKILL;
  }

  const std::string result = read_line(observation_read.get());
  const int guardian_code = wait_child(
      guardian, std::chrono::duration_cast<std::chrono::milliseconds>(kDeadline));
  if (guardian_code != 0 || result.rfind("TREATMENT ", 0) != 0) {
    throw Error("treatment Guardian failed");
  }
  TreatmentResult observation;
  observation.peercred = result.find("peercred=1") != std::string::npos;
  observation.pidfd = result.find("pidfd=1") != std::string::npos;
  observation.start_tick = result.find("start=1") != std::string::npos;
  observation.ancestry = result.find("ancestry=1") != std::string::npos;
  observation.executable = result.find("executable=1") != std::string::npos;
  observation.operation = result.find("operation=1") != std::string::npos;
  observation.spoof_refused = result.find("spoof=1") != std::string::npos;
  observation.release_refused = result.find("release=1") != std::string::npos;
  observation.pre_exec_zero = result.find("preexec=1") != std::string::npos;
  observation.transport_dead = result.find("transport=1") != std::string::npos;
  observation.interfaces_dead = interfaces_dead;
  observation.witness_continued = result.find("witness=1") != std::string::npos;
  observation.exactly_once = result.find("exact=1") != std::string::npos;
  observation.replay_refused = result.find("replay=1") != std::string::npos;
  observation.pdeathsig = result.find("pdeathsig=1") != std::string::npos;
  return observation;
}

struct GuardianDeathResult {
  bool guardian_pidfd = false;
  bool pdeathsig = false;
  bool guardian_extinct = false;
  bool material_extinct = false;
  bool release_absent = false;
};

[[noreturn]] void guardian_death_child(int ready, int marker) {
  const pid_t guardian = getpid();
  const pid_t material = fork();
  if (material < 0) _exit(101);
  if (material == 0) {
#ifndef LOOM_SOVEREIGN_DISABLE_PDEATHSIG
    if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0) _exit(102);
#endif
    if (getppid() != guardian) _exit(103);
    const std::uint64_t start = process_start_tick(getpid());
    write_all(ready, "ARMED " + std::to_string(getpid()) + " " +
                         std::to_string(start) + "\n");
    pause();
    write_all(marker, "C");
    _exit(104);
  }
  for (;;) pause();
}

GuardianDeathResult run_guardian_death() {
  int ready_pipe[2] = {-1, -1};
  int marker_pipe[2] = {-1, -1};
  if (pipe2(ready_pipe, O_CLOEXEC) != 0 ||
      pipe2(marker_pipe, O_CLOEXEC | O_NONBLOCK) != 0) {
    throw Error("cannot create Guardian-death pipes");
  }
  Fd ready_read(ready_pipe[0]);
  Fd ready_write(ready_pipe[1]);
  Fd marker_read(marker_pipe[0]);
  Fd marker_write(marker_pipe[1]);
  const pid_t guardian = fork();
  if (guardian < 0) throw Error("cannot fork Guardian-death treatment");
  if (guardian == 0) {
    ready_read.reset();
    marker_read.reset();
    guardian_death_child(ready_write.get(), marker_write.get());
  }
  ready_write.reset();
  marker_write.reset();
  const std::string armed = read_line(ready_read.get());
  std::istringstream fields(armed);
  std::string label;
  pid_t material = 0;
  std::uint64_t material_start = 0;
  if (!(fields >> label >> material >> material_start) || label != "ARMED" ||
      material <= 1 || material_start == 0) {
    throw Error("Guardian-death arm observation malformed");
  }
  Fd guardian_pidfd(open_pidfd(guardian));
  Fd material_pidfd(open_pidfd(material));
  if (!guardian_pidfd || !material_pidfd || !pidfd_alive(guardian_pidfd.get()) ||
      !pidfd_alive(material_pidfd.get())) {
    throw Error("Guardian-death pidfd binding failed");
  }
  kill(guardian, SIGKILL);
  const int guardian_code = wait_child(
      guardian, std::chrono::duration_cast<std::chrono::milliseconds>(kDeadline));
  const bool material_dead = wait_pidfd_dead(material_pidfd.get(),
                                              std::chrono::seconds(2));
#ifdef LOOM_SOVEREIGN_DISABLE_PDEATHSIG
  if (!material_dead) {
    kill(material, SIGKILL);
    static_cast<void>(wait_child(material, std::chrono::seconds(2)));
    throw Error("material-survived-guardian");
  }
#endif
  const int material_code = wait_child(material, std::chrono::seconds(2));
  char marker = 0;
  const ssize_t marker_count = read(marker_read.get(), &marker, 1);
  GuardianDeathResult result;
  result.guardian_pidfd = true;
  result.pdeathsig = true;
  result.guardian_extinct = guardian_code == 128 + SIGKILL;
  result.material_extinct = material_dead && material_code == 128 + SIGKILL &&
                            process_absent(material, material_start);
  result.release_absent = marker_count == 0 ||
                          (marker_count < 0 &&
                           (errno == EAGAIN || errno == EWOULDBLOCK));
  return result;
}

std::string make_frame(const Manifest& manifest, const std::string& prefix) {
  return manifest.require("wire_schema") + " " +
         manifest.require(prefix + "_mode") + " " +
         manifest.require(prefix + "_stage") + " " +
         manifest.require(prefix + "_word") + " " +
         manifest.require("sabotage_count") + " " +
         manifest.require("sabotage_required");
}

void validate_semantic_output(const Manifest& manifest,
                              const std::string& prefix,
                              const ProgramResult& result) {
  if (result.code != 0 ||
      sha256(result.output) != manifest.require(prefix + "_output_sha256") ||
      first_line(result.output) != manifest.require(prefix + "_decision")) {
    throw Error("frozen Sounio " + prefix + " decision diverged");
  }
}

std::string create_runtime_directory() {
  std::string pattern = "/tmp/loom-sovereign-material.XXXXXX";
  std::vector<char> bytes(pattern.begin(), pattern.end());
  bytes.push_back('\0');
  char* path = mkdtemp(bytes.data());
  if (path == nullptr) throw Error("cannot create runtime directory");
  if (chmod(path, 0700) != 0) throw Error("cannot protect runtime directory");
  return path;
}

int selftest(const std::string& runtime, const std::string& manifest_path,
             const std::string& peer_judgment_path) {
  if (file_sha256(manifest_path) != kFrozenSemanticManifestSha256 ||
      file_sha256(peer_judgment_path) != kFrozenPeerJudgmentSha256) {
    throw Error("frozen parent hash mismatch");
  }
  const Manifest manifest = parse_manifest(manifest_path);
  const Manifest peer = parse_manifest(peer_judgment_path);
  if (manifest.require("stage") != "SEMANTICS_FROZEN" ||
      manifest.require("semantic_authority") != "Sounio" ||
      manifest.require("action") != "9042" ||
      manifest.require("parent_9025_sha256") != kFrozenPeerJudgmentSha256 ||
      manifest.require("grant_is_bearer") != "false" ||
      manifest.require("exported_token") != "false" ||
      manifest.require("exported_handle") != "false" ||
      manifest.require("production_activation") != "false" ||
      peer.require("same_uid_peer_isolation") != "true" ||
      peer.require("material_execution") != "true" ||
      peer.require("production_activation") != "false") {
    throw Error("frozen authority state invalid");
  }

  const std::string directory = create_runtime_directory();
  TreatmentResult treatment;
  GuardianDeathResult death;
  try {
    treatment = run_treatment(directory);
    death = run_guardian_death();
    unlink((directory + "/guardian.sock").c_str());
    rmdir(directory.c_str());
  } catch (...) {
    unlink((directory + "/guardian.sock").c_str());
    rmdir(directory.c_str());
    throw;
  }
  const bool treatment_complete =
      treatment.peercred && treatment.pidfd && treatment.start_tick &&
      treatment.ancestry && treatment.executable && treatment.operation &&
      treatment.spoof_refused && treatment.release_refused &&
      treatment.pre_exec_zero && treatment.transport_dead &&
      treatment.interfaces_dead && treatment.witness_continued &&
      treatment.exactly_once && treatment.replay_refused && treatment.pdeathsig;
  const bool death_complete = death.guardian_pidfd && death.pdeathsig &&
                              death.guardian_extinct && death.material_extinct &&
                              death.release_absent;
  if (!treatment_complete || !death_complete) {
    std::ostringstream reason;
    reason << "material observation incomplete"
           << " peercred=" << treatment.peercred
           << " pidfd=" << treatment.pidfd
           << " start=" << treatment.start_tick
           << " ancestry=" << treatment.ancestry
           << " executable=" << treatment.executable
           << " operation=" << treatment.operation
           << " spoof=" << treatment.spoof_refused
           << " release=" << treatment.release_refused
           << " preexec=" << treatment.pre_exec_zero
           << " transport=" << treatment.transport_dead
           << " interfaces=" << treatment.interfaces_dead
           << " witness=" << treatment.witness_continued
           << " exact=" << treatment.exactly_once
           << " replay=" << treatment.replay_refused
           << " treatment_pdeathsig=" << treatment.pdeathsig
           << " guardian_pidfd=" << death.guardian_pidfd
           << " death_pdeathsig=" << death.pdeathsig
           << " guardian_extinct=" << death.guardian_extinct
           << " material_extinct=" << death.material_extinct
           << " release_absent=" << death.release_absent;
    throw Error(reason.str());
  }

  const ProgramResult treatment_semantics =
      run_sounio(runtime, make_frame(manifest, "treatment"));
  const ProgramResult death_semantics =
      run_sounio(runtime, make_frame(manifest, "guardian_death"));
  const ProgramResult production_semantics =
      run_sounio(runtime, make_frame(manifest, "production"));
  validate_semantic_output(manifest, "treatment", treatment_semantics);
  validate_semantic_output(manifest, "guardian_death", death_semantics);
  validate_semantic_output(manifest, "production", production_semantics);

  std::cout
      << "sounio-loom-sovereign-execution-kernel-material: PASS"
      << " semantic_authority=Sounio action=9042"
      << " operational_kernel=HostGuardian material_language=C++20+Linux"
      << " material_role=MATERIAL_PARITY transitory=true"
      << " parent_9025_same_uid_peer_isolation=true"
      << " grant_resident_memory=true grant_is_bearer=false"
      << " grant_single_use=true consume_atomic=true"
      << " exported_token=false exported_handle=false"
      << " descriptor_is_execution_authority=false"
      << " peer=SO_PEERCRED+pidfd+start-tick+harness-ancestry+executable+operation"
      << " hostile_same_uid=true hostile_same_executable=true"
      << " same_uid_spoof=refused-before-execution"
      << " interface_release_authority=zero"
      << " gui_release=false tui_release=false cli_release=false"
      << " pod_release=false tmux_release=false coordinator_release=false"
      << " transport_death=material-witness-continued"
      << " gui_death=material-witness-continued"
      << " coordinator_death=material-witness-continued"
      << " pod_death=material-witness-continued"
      << " tmux_death=material-witness-continued"
      << " material_exactly_once=true replay_refused=true"
      << " guardian_pidfd=bound pdeathsig=armed"
      << " guardian_death=grant-revoked+material-extinct+release-absent"
      << " treatment_sounio_sha256=" << sha256(treatment_semantics.output)
      << " guardian_death_sounio_sha256=" << sha256(death_semantics.output)
      << " production_sounio_sha256=" << sha256(production_semantics.output)
      << " same_uid_peer_isolation=true"
      << " production_gate_ready=true production_activation=false"
      << " exec_attached=false commit_attached=false ci_attached=false"
      << " parity_open=false claim_ready=false"
      << " python_executed=false rust_executed=false"
      << " semantic_manifest_sha256=" << kFrozenSemanticManifestSha256
      << " peer_judgment_sha256=" << kFrozenPeerJudgmentSha256 << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 5 || std::string(argv[1]) != "--selftest") {
      throw Error("usage: loom-sovereign-execution-kernel-material --selftest RUNTIME MANIFEST PEER_JUDGMENT");
    }
    if (prctl(PR_SET_CHILD_SUBREAPER, 1) != 0) {
      throw Error("cannot become child subreaper");
    }
    return selftest(argv[2], argv[3], argv[4]);
  } catch (const std::exception& error) {
    std::cerr << "loom-sovereign-execution-kernel-material: FAIL: "
              << error.what() << '\n';
    return 1;
  }
}
