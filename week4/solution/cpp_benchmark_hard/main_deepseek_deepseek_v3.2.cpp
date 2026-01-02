
#include <iostream>
#include <chrono>
#include <cstdint>
#include <vector>
#include <limits>
#include <algorithm>

using namespace std;
using namespace std::chrono;

class LCG {
private:
    uint32_t state;
    const uint32_t a = 1664525;
    const uint32_t c = 1013904223;
    const uint32_t m = 0xFFFFFFFF; // 2^32 - 1
    
public:
    explicit LCG(uint32_t seed) : state(seed) {}
    
    uint32_t next() {
        state = static_cast<uint32_t>((static_cast<uint64_t>(a) * state + c) & 0xFFFFFFFFULL);
        return state;
    }
};

int64_t max_subarray_sum(int n, uint32_t seed, int min_val, int max_val) {
    LCG gen(seed);
    int range = max_val - min_val + 1;
    
    // Generate numbers on the fly to avoid memory allocation
    int64_t current_sum = 0;
    int64_t max_sum = numeric_limits<int64_t>::min();
    
    for (int i = 0; i < n; ++i) {
        current_sum = 0;
        LCG inner_gen(seed);  // Reset generator for each starting point
        
        // Skip to position i
        for (int skip = 0; skip <= i; ++skip) {
            inner_gen.next();
        }
        
        for (int j = i; j < n; ++j) {
            uint32_t val = inner_gen.next();
            int random_num = static_cast<int>(val % range) + min_val;
            current_sum += random_num;
            if (current_sum > max_sum) {
                max_sum = current_sum;
            }
        }
    }
    
    return max_sum;
}

int64_t total_max_subarray_sum(int n, uint32_t initial_seed, int min_val, int max_val) {
    LCG outer_gen(initial_seed);
    int64_t total_sum = 0;
    
    for (int run = 0; run < 20; ++run) {
        uint32_t seed = outer_gen.next();
        total_sum += max_subarray_sum(n, seed, min_val, max_val);
    }
    
    return total_sum;
}

int main() {
    const int n = 10000;
    const uint32_t initial_seed = 42;
    const int min_val = -10;
    const int max_val = 10;
    
    auto start_time = high_resolution_clock::now();
    int64_t result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    auto end_time = high_resolution_clock::now();
    
    double elapsed_time = duration<double>(end_time - start_time).count();
    
    cout << "Total Maximum Subarray Sum (20 runs): " << result << endl;
    cout << "Execution Time: " << fixed << elapsed_time << " seconds" << endl;
    
    return 0;
}
