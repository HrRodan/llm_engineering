
#include <bits/stdc++.h>

using namespace std;

struct LCG {
    uint32_t value;
    static constexpr uint32_t A = 1664525;
    static constexpr uint32_t C = 1013904223;
    LCG(uint32_t seed) : value(seed) {}
    uint32_t operator()() {
        value = A * value + C;
        return value;
    }
};

long long max_subarray_sum(int n, uint64_t seed, int min_val, int max_val) {
    LCG gen(seed);
    vector<int> arr(n);
    int range = max_val - min_val + 1;
    for (int i = 0; i < n; i++) {
        arr[i] = (gen() % range) + min_val;
    }
    // Kadane's algorithm
    long long max_sum = arr[0];
    long long curr = arr[0];
    for (int i = 1; i < n; i++) {
        curr = max((long long)arr[i], curr + arr[i]);
        max_sum = max(max_sum, curr);
    }
    return max_sum;
}

long long total_max_subarray_sum(int n, uint64_t initial_seed, int min_val, int max_val) {
    long long total = 0;
    LCG gen(initial_seed);
    for (int i = 0; i < 20; i++) {
        uint64_t seed = gen();
        total += max_subarray_sum(n, seed, min_val, max_val);
    }
    return total;
}

int main() {
    int n = 10000;
    uint64_t initial_seed = 42;
    int min_val = -10;
    int max_val = 10;

    auto start = chrono::high_resolution_clock::now();
    long long result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Total Maximum Subarray Sum (20 runs): " << result << endl;
    cout << "Execution Time: " << fixed << setprecision(6) << elapsed.count() << " seconds" << endl;
    return 0;
}
