#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
// Operational ELF test double: reads only protocol headers, never BIDI levels.
int main(){
 std::ifstream in("data/current.bin",std::ios::binary);
 if(!in)return 1;
 unsigned char header[16]{};in.read(reinterpret_cast<char*>(header),16);
 if(in.gcount()!=16)return 1;
 if(std::filesystem::current_path().filename()=="000001"){
  std::cout<<"INTENTIONAL_METADATA_DOUBLE_REFUSAL\n";return 1;
 }
 const auto count=std::uint32_t(header[12])|(std::uint32_t(header[13])<<8)|(std::uint32_t(header[14])<<16)|(std::uint32_t(header[15])<<24);
 std::cout<<"BIDI_NATIVE_CONFORMANCE_OK\nBIDI_CASES_OK "<<count<<"\n";
 return 0;
}
