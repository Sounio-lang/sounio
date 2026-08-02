#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#define main cs6_hapg_full53_kernel_main
#include "cs6_affine_projective_cocycle_full53_probe.cpp"
#undef main

namespace {

void replace_once(std::string& text, const std::string& from,
                  const std::string& to) {
  const std::size_t position = text.find(from);
  if (position == std::string::npos ||
      text.find(from, position + from.size()) != std::string::npos) {
    throw std::runtime_error("H-APG kernel metadata replacement mismatch");
  }
  text.replace(position, from.size(), to);
}

}  // namespace

int main(int argc, char** argv) {
  std::ostringstream captured;
  std::streambuf* original = std::cout.rdbuf(captured.rdbuf());
  const int result = cs6_hapg_full53_kernel_main(argc, argv);
  std::cout.flush();
  std::cout.rdbuf(original);

  std::string receipt = captured.str();
  if (result == 0) {
    try {
      replace_once(
          receipt,
          "SCHEMA=sounio.cs6.affine-projective-cocycle-full53-leaf.v1\n",
          "SCHEMA=sounio.cs6.hapg-full-source-cover-leaf.v1\n");
      replace_once(
          receipt,
          "EXECUTION_SCOPE=FIFTY_TWO_PARENT_COMPUTABLE_DYADIC_LEAF_HAPG_CAPD_CPU_CORPUS\n",
          "EXECUTION_SCOPE=ARBITRARY_MANIFEST_BOUND_DYADIC_LEAF_HAPG_CAPD_CPU\n");
    } catch (const std::exception& error) {
      std::cerr << "cover worker error: " << error.what() << '\n';
      return 1;
    }
  }
  std::cout << receipt;
  return result;
}
