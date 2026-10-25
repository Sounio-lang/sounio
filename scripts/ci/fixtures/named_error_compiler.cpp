// Deliberately failing compiler fixture, not a substitute for a runtime test.
#include <iostream>
#include <string_view>
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--version") {
        std::cout << "Madaros v0.80.0\n";
        return 0;
    }
    if (argc > 1 && std::string_view(argv[1]) == "check") {
        std::cout << "error[E-USE-MOD-CAP]: deliberately injected diagnostic\n";
        return 1;
    }
    return 2;
}
