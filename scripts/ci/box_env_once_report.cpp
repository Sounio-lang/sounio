#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// These are cumulative counters, not per-module deltas or resident bytes.
static void verify(std::istream& input) {
    std::array<long long, 3> calls{}, refill{};
    std::string line;
    long long group_module = 0;
    int next = 0;
    bool seen = false;
    while (std::getline(input, line)) {
        if (!line.starts_with("box_env_alloc:")) continue;
        long long module = 0, site = 0, count = 0, bytes = 0;
        int end = 0;
        if (std::sscanf(line.c_str(),
            "box_env_alloc: module=%lld site=%lld calls=%lld pool_refill=%lld%n",
            &module, &site, &count, &bytes, &end) != 4 ||
            line.substr(static_cast<std::size_t>(end)).find_first_not_of(" \t\r") != std::string::npos)
            throw std::runtime_error("malformed BOX environment counter");
        if (site != next || count < 0 || bytes < 0)
            throw std::runtime_error("invalid BOX environment counter order/value");
        if (next == 0) group_module = module;
        if (module != group_module)
            throw std::runtime_error("incomplete BOX environment module");
        const auto index = static_cast<std::size_t>(site);
        if (count < calls[index] || bytes < refill[index])
            throw std::runtime_error("decreasing BOX environment counters");
        calls[index] = count;
        refill[index] = bytes;
        next = (next + 1) % 3;
        seen = true;
    }
    if (!seen || next != 0)
        throw std::runtime_error("missing/incomplete BOX environment counters");
    // Avoid overflow even for corrupt diagnostic input.
    if (calls[0] > 1 || calls[1] > 1 || calls[2] > 1 ||
        calls[0] + calls[1] + calls[2] != 1)
        throw std::runtime_error("BOX_ENV_NOT_ONCE");
}

static std::string fixture(int reads) {
    return "box_env_alloc: module=0 site=0 calls=" + std::to_string(reads) +
        " pool_refill=0\nbox_env_alloc: module=0 site=1 calls=0 pool_refill=0\n"
        "box_env_alloc: module=0 site=2 calls=0 pool_refill=0\n";
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            std::istringstream positive(fixture(1));
            verify(positive);
            for (const auto& bad : {fixture(0), fixture(6), std::string{},
                    std::string("box_env_alloc: module=0 site=0 calls=1 pool_refill=0\n"),
                    fixture(1) + fixture(0)}) {
                bool refused = false;
                try { std::istringstream input(bad); verify(input); }
                catch (const std::runtime_error&) { refused = true; }
                if (!refused) throw std::runtime_error("negative control accepted");
            }
            std::cout << "BOX_ENV_REPORT_CONTROLS_OK\n";
        } else {
            if (argc != 2) throw std::runtime_error("usage: box_env_once_report LOG|--self-test");
            std::ifstream input(argv[1]);
            if (!input) throw std::runtime_error("cannot open diagnostic log");
            verify(input);
            std::cout << "BOX_ENV_READ_ONCE_OK\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
