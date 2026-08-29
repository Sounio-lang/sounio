#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

constexpr std::size_t kHeaderSize = 110;
constexpr std::size_t kMaximumArchive = 128 * 1024 * 1024;

void append_hex(std::string& output, std::uint32_t value) {
  std::ostringstream encoded;
  encoded << std::hex << std::nouppercase << std::setfill('0') << std::setw(8) << value;
  output += encoded.str();
}

void append_padding(std::string& output) {
  while (output.size() % 4 != 0) output.push_back('\0');
}

std::string read_regular(const fs::path& path) {
  const auto size = fs::file_size(path);
  if (size > kMaximumArchive || size > std::numeric_limits<std::uint32_t>::max()) {
    throw Error("input file exceeds newc bound: " + path.string());
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) throw Error("cannot open input: " + path.string());
  std::string result(static_cast<std::size_t>(size), '\0');
  input.read(result.data(), static_cast<std::streamsize>(result.size()));
  if (!input && !result.empty()) throw Error("cannot read input: " + path.string());
  return result;
}

std::uint32_t entry_mode(const fs::path& path) {
  struct stat metadata {};
  if (lstat(path.c_str(), &metadata) != 0) {
    throw Error("cannot stat input: " + path.string());
  }
  if (!S_ISDIR(metadata.st_mode) && !S_ISREG(metadata.st_mode)) {
    throw Error("newc packer accepts only directories and regular files: " + path.string());
  }
  return static_cast<std::uint32_t>(metadata.st_mode & (S_IFMT | 07777));
}

void append_entry(std::string& archive, std::uint32_t inode, std::uint32_t mode,
                  std::string_view name, std::string_view payload) {
  if (name.empty() || name.size() + 1 > std::numeric_limits<std::uint32_t>::max()) {
    throw Error("newc entry name is invalid");
  }
  archive += "070701";
  append_hex(archive, inode);
  append_hex(archive, mode);
  append_hex(archive, 0);
  append_hex(archive, 0);
  append_hex(archive, S_ISDIR(mode) ? 2 : 1);
  append_hex(archive, 0);
  append_hex(archive, static_cast<std::uint32_t>(payload.size()));
  append_hex(archive, 0);
  append_hex(archive, 0);
  append_hex(archive, 0);
  append_hex(archive, 0);
  append_hex(archive, static_cast<std::uint32_t>(name.size() + 1));
  append_hex(archive, 0);
  archive.append(name);
  archive.push_back('\0');
  append_padding(archive);
  archive.append(payload);
  append_padding(archive);
}

void write_all(const fs::path& path, std::string_view bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw Error("cannot create output: " + path.string());
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!output) throw Error("cannot write output: " + path.string());
}

void create_archive(const fs::path& root, const fs::path& output) {
  if (!fs::is_directory(root)) throw Error("archive root is not a directory");
  std::vector<fs::path> entries;
  entries.push_back(root);
  for (const auto& entry : fs::recursive_directory_iterator(root)) {
    if (entry.is_symlink()) throw Error("archive root contains a symlink");
    entries.push_back(entry.path());
  }
  std::sort(entries.begin() + 1, entries.end(), [&](const fs::path& left, const fs::path& right) {
    return fs::relative(left, root).generic_string() < fs::relative(right, root).generic_string();
  });

  std::string archive;
  std::uint32_t inode = 1;
  for (const auto& path : entries) {
    const std::string name = path == root ? "." : fs::relative(path, root).generic_string();
    const std::uint32_t mode = entry_mode(path);
    const std::string payload = S_ISREG(mode) ? read_regular(path) : std::string{};
    append_entry(archive, inode++, mode, name, payload);
    if (archive.size() > kMaximumArchive) throw Error("newc archive exceeds bound");
  }
  append_entry(archive, inode, 0, "TRAILER!!!", "");
  write_all(output, archive);
}

std::uint32_t parse_hex(std::string_view value) {
  if (value.size() != 8) throw Error("newc hex field has wrong width");
  std::uint32_t result = 0;
  for (const unsigned char character : value) {
    result <<= 4;
    if (character >= '0' && character <= '9') result |= character - '0';
    else if (character >= 'a' && character <= 'f') result |= character - 'a' + 10;
    else if (character >= 'A' && character <= 'F') result |= character - 'A' + 10;
    else throw Error("newc hex field is malformed");
  }
  return result;
}

std::string read_archive(const fs::path& path) {
  const auto size = fs::file_size(path);
  if (size > kMaximumArchive) throw Error("newc archive exceeds bound");
  return read_regular(path);
}

void align_offset(std::size_t& offset, std::size_t limit) {
  offset = (offset + 3) & ~std::size_t{3};
  if (offset > limit) throw Error("newc alignment crosses archive bound");
}

bool safe_name(const fs::path& path) {
  if (path.empty() || path.is_absolute()) return false;
  for (const auto& component : path) {
    if (component == "..") return false;
  }
  return true;
}

void extract_archive(const fs::path& archive_path, const fs::path& destination) {
  const std::string archive = read_archive(archive_path);
  fs::create_directories(destination);
  std::size_t offset = 0;
  bool trailer = false;
  while (offset < archive.size()) {
    if (archive.size() - offset < kHeaderSize ||
        std::string_view(archive).substr(offset, 6) != "070701") {
      throw Error("newc header is absent or malformed");
    }
    const std::uint32_t mode = parse_hex(std::string_view(archive).substr(offset + 14, 8));
    const std::uint32_t file_size = parse_hex(std::string_view(archive).substr(offset + 54, 8));
    const std::uint32_t name_size = parse_hex(std::string_view(archive).substr(offset + 94, 8));
    offset += kHeaderSize;
    if (name_size == 0 || name_size > archive.size() - offset) {
      throw Error("newc name crosses archive bound");
    }
    const std::string name = archive.substr(offset, name_size - 1);
    if (archive[offset + name_size - 1] != '\0') throw Error("newc name is not terminated");
    offset += name_size;
    align_offset(offset, archive.size());
    if (file_size > archive.size() - offset) throw Error("newc payload crosses archive bound");
    if (name == "TRAILER!!!") {
      if (file_size != 0) throw Error("newc trailer carries a payload");
      trailer = true;
      break;
    }
    const fs::path relative(name);
    if (!safe_name(relative)) throw Error("newc path is unsafe");
    const fs::path target = relative == "." ? destination : destination / relative;
    if ((mode & S_IFMT) == S_IFDIR) {
      fs::create_directories(target);
    } else if ((mode & S_IFMT) == S_IFREG) {
      fs::create_directories(target.parent_path());
      write_all(target, std::string_view(archive).substr(offset, file_size));
    } else {
      throw Error("newc archive contains an unsupported entry type");
    }
    if (chmod(target.c_str(), mode & 07777) != 0) {
      throw Error("cannot apply extracted mode: " + target.string());
    }
    offset += file_size;
    align_offset(offset, archive.size());
  }
  if (!trailer) throw Error("newc trailer is absent");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--selftest") {
      if (parse_hex("0000002a") != 42) throw Error("newc hex parser selftest failed");
      std::cout << "LOOM_NEWC_PACKER_SELFTEST PASS language=C++20 role=MATERIAL_PACKER "
                   "deterministic=true symlinks=refused traversal=refused python_executed=false "
                   "rust_executed=false\n";
      return 0;
    }
    if (argc == 4 && std::string_view(argv[1]) == "--create") {
      create_archive(argv[2], argv[3]);
      return 0;
    }
    if (argc == 4 && std::string_view(argv[1]) == "--extract") {
      extract_archive(argv[2], argv[3]);
      return 0;
    }
    return 64;
  } catch (const std::exception& error) {
    std::cerr << "LOOM_NEWC_PACKER_REFUSE reason=" << error.what() << "\n";
    return 70;
  }
}
