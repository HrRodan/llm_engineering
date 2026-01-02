
#include <iostream>
#include <iomanip>
#include <chrono>

// Compute the series result.
static inline double calculate(long long iterations, double param1, double param2) {
    double result = 1.0;
    // Initialize j1 and j2 for i = 1.
    double j1 = param1 - param2;
    double j2 = param1 + param2;
#pragma GCC unroll 8
    for (long long i = 1; i <= iterations; ++i) {
        result -= 1.0 / j1;
        result += 1.0 / j2;
        j1 += param1;
        j2 += param1;
    }
    return result;
}

int main() {
    long long iterations = 200000000LL;
    double param1 = 4.0;
    double param2 = 1.0;

    auto start = std::chrono::steady_clock::now();
    double result = calculate(iterations, param1, param2) * 4.0;
    auto end = std::chrono::steady_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "Result: " << std::fixed << std::setprecision(12) << result << '\n';
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) << elapsed << " seconds\n";
    return 0;
}
