#include <iomanip>
#include <iostream>
#include <string_view>
int main(int argc, char** argv) {
    if (argc != 2) return 64;
    std::cout << std::fixed << std::setprecision(6);
    if (std::string_view(argv[1]) == "private") {
        std::cout << "PRIVATE_DIRECT_OK\n" << 1.25 << '\n' << 2.5 << "\nSYNTH_PRIVATE_PRINT_OK\n";
    } else if (std::string_view(argv[1]) == "visible") {
        std::cout << "VISIBLE_OVERRIDE_OK\n" << 1.25 << "\nSYNTH_VISIBLE_PRINT_OK\n";
    } else return 64;
}
