#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
int main() {
    const char* p = std::getenv("WAVE3_ENV_PRESENT");
    const char* e = std::getenv("WAVE3_ENV_EMPTY");
    const std::string maximum(63, 'Z');
    const char* m = std::getenv(maximum.c_str());
    const bool ok = p && std::strcmp(p, "visible") == 0 && e && *e == '\0' &&
        !std::getenv("WAVE3_ENV_MISSING_7319") && !std::getenv("WAVE3_ENV_PREFIX") &&
        !std::getenv("wave3_env_present") && !std::getenv("") &&
        !std::getenv("WAVE3_ENV_PRESENT=visible") && m && std::strcmp(m, "max") == 0 &&
        std::getenv("PATH");
    if (!ok) return 1;
    std::cout << "GETENV_CPP_REFERENCE_OK\n";
}
