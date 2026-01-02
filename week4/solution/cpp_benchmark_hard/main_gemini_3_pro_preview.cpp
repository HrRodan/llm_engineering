
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdint>

// LCG parameters as defined in the Python code
constexpr uint32_t A = 1664525;
constexpr uint32_t C = 1013904223;

struct LCG {
    uint32_t value;
    LCG(uint32_t seed) : value(seed) {}
    
    // Updates state and returns the new value (matching Python generator behavior)
    inline uint32_t next() {
        value = A * value + C; // Implicit modulo 2^32 via overflow
        return value;
    }
};

// Computes the maximum subarray sum using Kadane's Algorithm.
// This is O(N) compared to the O(N^2) brute force in the Python script.
// It produces identical outputs for the Maximum Subarray Sum problem.
long long max_subarray_sum(int n, uint32_t seed, int min_val, int max_val) {
    LCG lcg(seed);
    // Determine the range for modulo operation
    uint32_t range = static_cast<uint32_t>(max_val - min_val + 1);

    // Generate the first number
    // Important: cast to long long before adding min_val to avoid unsigned arithmetic issues if min_val is negative
    long long val = static_cast<long long>(lcg.next() % range) + min_val;
    
    long long max_so_far = val;
    long long current_max = val;

    for (int i = 1; i < n; ++i) {
        val = static_cast<long long>(lcg.next() % range) + min_val;
        
        // Kadane's logic: if adding the current element to previous max is worse than starting fresh, start fresh.
        if (current_max > 0) {
            current_max += val;
        } else {
            current_max = val;
        }

        if (current_max > max_so_far) {
            max_so_far = current_max;
        }
    }
    return max_so_far;
}

long long total_max_subarray_sum(int n, uint32_t initial_seed, int min_val, int max_val) {
    long long total_sum = 0;
    LCG master_lcg(initial_seed);
    
    for (int i = 0; i < 20; ++i) {
        // Generate seed for this run using the master LCG
        uint32_t seed = master_lcg.next();
        total_sum += max_subarray_sum(n, seed, min_val, max_val);
    }
    return total_sum;
}

int main() {
    // Parameters
    const int n = 10000;
    const uint32_t initial_seed = 42;
    const int min_val = -10;
    const int max_val = 10;

    auto start_time = std::chrono::steady_clock::now();
    
    long long result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Total Maximum Subarray Sum (20 runs): " << result << "\n";
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) << elapsed.count() << " seconds" << std::endl;

    return 0;
}
