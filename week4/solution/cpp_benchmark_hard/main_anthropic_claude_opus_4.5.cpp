
#include <iostream>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <limits>

int64_t max_subarray_sum(int n, uint32_t seed, int min_val, int max_val) {
    uint32_t value = seed;
    uint32_t range = static_cast<uint32_t>(max_val - min_val + 1);
    
    // Using Kadane's algorithm - O(n) instead of O(n²)
    int64_t max_sum = std::numeric_limits<int64_t>::min();
    int64_t current_sum = 0;
    
    for (int i = 0; i < n; ++i) {
        value = 1664525u * value + 1013904223u;
        int64_t val = static_cast<int64_t>(value % range) + min_val;
        current_sum = std::max(val, current_sum + val);
        max_sum = std::max(max_sum, current_sum);
    }
    
    return max_sum;
}

int64_t total_max_subarray_sum(int n, uint32_t initial_seed, int min_val, int max_val) {
    int64_t total_sum = 0;
    uint32_t value = initial_seed;
    
    for (int i = 0; i < 20; ++i) {
        value = 1664525u * value + 1013904223u;
        total_sum += max_subarray_sum(n, value, min_val, max_val);
    }
    
    return total_sum;
}

int main() {
    int n = 10000;
    uint32_t initial_seed = 42;
    int min_val = -10;
    int max_val = 10;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    int64_t result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "Total Maximum Subarray Sum (20 runs): " << result << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Execution Time: " << elapsed.count() << " seconds" << std::endl;
    
    return 0;
}
