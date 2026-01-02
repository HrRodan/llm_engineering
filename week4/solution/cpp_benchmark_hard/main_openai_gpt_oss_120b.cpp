#include <bits/stdc++.h>
#include <boost/multiprecision/cpp_int.hpp>

using namespace std;
using boost::multiprecision::cpp_int;

static constexpr uint32_t A = 1664525u;
static constexpr uint32_t C = 1013904223u;

cpp_int max_subarray_sum(int n, uint32_t seed, int64_t min_val, int64_t max_val) {
    uint32_t state = seed;
    uint64_t range = static_cast<uint64_t>(max_val - min_val) + 1ULL;

    bool first = true;
    cpp_int cur = 0, best = 0;

    for (int i = 0; i < n; ++i) {
        state = state * A + C;                     // LCG step, overflow wraps modulo 2^32
        uint32_t rv = state;
        uint64_t mod_part = static_cast<uint64_t>(rv) % range;
        cpp_int x = cpp_int(mod_part) + min_val;   // value in [min_val, max_val]

        if (first) {
            cur = x;
            best = x;
            first = false;
        } else {
            cpp_int cand = cur + x;
            cur = (cand > x) ? cand : x;
            if (cur > best) best = cur;
        }
    }
    return best;
}

int main() {
    // Parameters (hard‑coded as in the original Python script)
    const int n = 10000;
    const uint32_t initial_seed = 42u;
    const int64_t min_val = -10;
    const int64_t max_val = 10;

    // Timing start
    auto start = chrono::high_resolution_clock::now();

    cpp_int total_sum = 0;
    uint32_t outer_state = initial_seed;

    for (int run = 0; run < 20; ++run) {
        outer_state = outer_state * A + C;                 // generate next seed
        cpp_int run_max = max_subarray_sum(n, outer_state, min_val, max_val);
        total_sum += run_max;
    }

    // Timing end
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    // Output
    cout << "Total Maximum Subarray Sum (20 runs): " << total_sum << '\n';
    cout << "Execution Time: " << fixed << setprecision(6) << elapsed.count() << " seconds\n";

    return 0;
}
