// Independent binary concordance for the real Sounio write/append fixture.
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
int main() {
 const auto read=[](const char* p) {
  std::ifstream input(p,std::ios::binary);
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input),{});
 };
 const auto bytes=read("/tmp/stdlib_append_file_checked.bin");
 const std::vector<std::uint8_t> expected{0x41,0x00,0xff,0x42,0x43};
 if(bytes!=expected) {std::cerr<<"binary append bytes differ\n";return 1;}
 if(read("/tmp/stdlib_append_file_truncate.bin")!=std::vector<std::uint8_t>{0x42}) {
  std::cerr<<"truncate bytes differ\n";return 2;
 }
 std::cout<<"APPEND_FILE_BYTES_OK\n";
}
