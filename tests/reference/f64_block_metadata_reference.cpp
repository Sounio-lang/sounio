#include <tuple>
#include <iostream>

int main() {
    for (bool flag : {false, true}) {
        auto tuple_result = flag
            ? [] { auto t = std::tuple{7, 2.0}; return std::get<1>(t); }()
            : [] { auto t = std::tuple{9, 2.0}; return std::get<1>(t); }();
        const double x = 2.0;
        auto ref_result = flag
            ? [&] { const double* p = &x; return *p; }()
            : [&] { const double* q = &x; return *q; }();
        if (tuple_result != 2.0 || ref_result != 2.0) return 1;
        if (tuple_result / 16.0 != 0.125 || ref_result / 16.0 != 0.125) return 2;
    }
    std::cout << "CXX23_BLOCK_METADATA_REFERENCE_OK expected_ratio=0.125\n";
}
