#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) return 1;
    std::string source(std::istreambuf_iterator<char>{input}, {});
    if (source.find(")SOUNIO_SF64\"") != std::string::npos) return 1;
    std::cout << "static constexpr char SOUNIO_SF64_SOURCE[] = R\"SOUNIO_SF64("
              << source << ")SOUNIO_SF64\";\n";
}
