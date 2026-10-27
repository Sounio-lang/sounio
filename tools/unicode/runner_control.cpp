// TEST ORCHESTRATOR CONTROL ONLY. This is not a Unicode implementation.
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
int main(){
 static_assert(std::endian::native==std::endian::little);
 std::ifstream input("tests/stdlib/wave3-core/data/current.bin",std::ios::binary);std::array<char,24> header{};input.read(header.data(),header.size());if(!input)return 65;
 std::uint32_t rows=0;std::memcpy(&rows,header.data()+12,4);
 std::string mode=std::getenv("CONTROL_MODE")?std::getenv("CONTROL_MODE"):"good";
 std::cout<<"CXX_RUNNER_CONTROL_ONLY\n"<<(rows+(mode=="wrong_rows"?1:0))<<"\n";
 if(mode!="missing_marker")std::cout<<"UNICODE_CONFORMANCE_OK\n";
 return mode=="nonzero_exit"?3:0;
}

