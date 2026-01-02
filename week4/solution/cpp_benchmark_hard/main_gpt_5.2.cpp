
#include <bits/stdc++.h>
using namespace std;

static inline uint32_t lcg_next(uint32_t &v) {
    v = v * 1664525u + 1013904223u; // mod 2^32 via uint32 overflow
    return v;
}

static inline long long max_subarray_sum_kadane(int n, uint32_t seed, int min_val, int max_val) {
    const uint32_t range = (uint32_t)(max_val - min_val + 1);
    uint32_t v = seed;

    // Generate first value
    uint32_t r = lcg_next(v);
    long long x = (long long)(int)(r % range) + (long long)min_val;

    long long best = x;
    long long cur = x;

    for (int i = 1; i < n; ++i) {
        r = lcg_next(v);
        x = (long long)(int)(r % range) + (long long)min_val;

        long long cur_plus = cur + x;
        cur = (cur_plus > x) ? cur_plus : x;
        best = (best > cur) ? best : cur;
    }
    return best;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const int n = 10000;
    const uint32_t initial_seed = 42u;
    const int min_val = -10;
    const int max_val = 10;

    auto t0 = chrono::steady_clock::now();

    long long total_sum = 0;
    uint32_t outer = initial_seed;
    for (int k = 0; k < 20; ++k) {
        uint32_t seed = lcg_next(outer);
        total_sum += max_subarray_sum_kadane(n, seed, min_val, max_val);
    }

    auto t1 = chrono::steady_clock::now();
    double seconds = chrono::duration<double>(t1 - t0).count();

    cout << "Total Maximum Subarray Sum (20 runs): " << total_sum << "\n";
    cout.setf(std::ios::fixed);
    cout << "Execution Time: " << setprecision(6) << seconds << " seconds\n";

    return 0;
}
