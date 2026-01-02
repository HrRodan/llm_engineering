#include <bits/stdc++.h>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Configuration
    constexpr int n = 10000;
    constexpr uint32_t initial_seed = 42;
    constexpr int min_val = -10;
    constexpr int max_val = 10;
    constexpr uint32_t a = 1664525;
    constexpr uint32_t c = 1013904223;
    constexpr int runs = 20;

    const int64_t range = static_cast<int64_t>(max_val) - static_cast<int64_t>(min_val) + 1;

    auto max_subarray_sum = [&](uint32_t seed) -> int64_t {
        uint32_t state = seed;
        int64_t cur_sum = 0;
        int64_t max_sum = std::numeric_limits<int64_t>::min();
        for (int i = 0; i < n; ++i) {
            // Advance LCG (modulo 2^32 automatically with unsigned arithmetic)
            state = a * state + c;
            int64_t val = static_cast<int64_t>(state % range) + min_val;
            cur_sum = std::max(val, cur_sum + val);
            max_sum = std::max(max_sum, cur_sum);
        }
        return max_sum;
    };

    auto start = std::chrono::high_resolution_clock::now();

    uint32_t master_state = initial_seed;
    int64_t total_sum = 0;
    for (int i = 0; i < runs; ++i) {
        master_state = a * master_state + c; // generate next seed
        total_sum += max_subarray_sum(master_state);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "Total Maximum Subarray Sum (20 runs): " << total_sum << '\n';
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) << elapsed << " seconds\n";

    return 0;
}
