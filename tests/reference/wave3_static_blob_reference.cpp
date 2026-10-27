#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
int main(int argc,char**argv) {
    if(argc!=2)return 64;
    std::filesystem::path root(argv[1]);
    std::filesystem::create_directories(root/"nested");
    const std::array<unsigned char,5> payload{0,1,127,128,255};
    std::ofstream file(root/"nested"/"payload.bin",std::ios::binary);
    file.write(reinterpret_cast<const char*>(payload.data()),payload.size());
    file.close();
    std::ofstream(root/"empty.bin",std::ios::binary).close();
    if(!file)return 1;
    std::cout<<"STATIC_BLOB_REFERENCE_CREATED\n";
}
