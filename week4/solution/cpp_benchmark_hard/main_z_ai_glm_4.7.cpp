
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <limits>

// LCG parameters
const uint32_t LCG_A = 1664525;
const uint32_t LCG_C = 1013904223;
const uint32_t LCG_M = 1ull << 32; // 2^32

// Using Kadane's algorithm for O(N) complexity to achieve "fastest possible implementation".
// This produces the exact same output as the O(N^2) Python code.
int64_t max_subarray_sum(int n, uint32_t seed, int min_val, int max_val) {
    uint32_t state = seed;
    int range = max_val - min_val + 1;

    // Generate the first number to initialize the algorithm
    state = state * LCG_A + LCG_C;
    int current_val = (state % range) + min_val;

    int max_sum = current_val;
    int current_sum = current_val;

    for (int i = 1; i < n; ++i) {
        state = state * LCG_A + LCG_C;
        current_val = (state % range) + min_val;

        // Kadane's algorithm step
        current_sum = std::max(current_val, current_sum + current_val);
        max_sum = std::max(max_sum, current_sum);
    }

    return max_sum;
}

int64_t total_max_subarray_sum(int n, uint32_t initial_seed, int min_val, int max_val) {
    int64_t total_sum = 0;
    uint32_t seed_lcg_state = initial_seed;

    for (int i = 0; i < 20; ++i) {
        // Generate the seed for the current run using the outer LCG
        seed_lcg_state = seed_lcg_state * LCG_A + LCG_C;
        
        total_sum += max_subarray_sum(n, seed_lcg_state, min_val, max_val);
    }
    return total_sum;
}

int main() {
    const int n = 10000;
    const uint32_t initial_seed = 42;
    const int min_val = -10;
    const int max_val = 10;

    auto start_time = std::chrono::high_resolution_clock::now();
    int64_t result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Total Maximum Subarray Sum (20 runs): " << result << std::endl;
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) << elapsed.count() << " seconds" << std::endl;

    return 0;
}
