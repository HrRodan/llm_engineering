
#include <cstdio>
#include <chrono>
#include <cstdint>

static inline double calculate_200m() {
    const uint32_t iterations = 200000000u;

    double result = 1.0;

    // i=1 => (4*i-1)=3, (4*i+1)=5
    double dminus = 3.0;
    double dplus  = 5.0;

    uint32_t i = 1;
    for (; i + 7u <= iterations; i += 8u) {
        result -= 1.0 / dminus; result += 1.0 / dplus; dminus += 4.0; dplus += 4.0;
        result -= 1.0 / dminus; result += 1.0 / dplus; dminus += 4.0; dplus += 4.0;
        result -= 1.0 / dminus; result += 1.0 / dplus; dminus += 4.0; dplus += 4.0;
        result -= 1.0 / dminus; result += 1.0 / dplus; dminus += 4.0; dplus += 4.0;
        result -= 1.0 / dminus; result += 1.0 / dplus; dminus += 4.0; dplus += 4.0;
        result -= 1.0 / dminus; result += 1.0 / dplus; dminus += 4.0; dplus += 4.0;
        result -= 1.0 / dminus; result += 1.0 / dplus; dminus += 4.0; dplus += 4.0;
        result -= 1.0 / dminus; result += 1.0 / dplus; dminus += 4.0; dplus += 4.0;
    }
    for (; i <= iterations; ++i) {
        result -= 1.0 / dminus;
        result += 1.0 / dplus;
        dminus += 4.0;
        dplus += 4.0;
    }

    return result;
}

int main() {
    using clock = std::chrono::steady_clock;

    const auto start = clock::now();
    const double result = calculate_200m() * 4.0;
    const auto end = clock::now();

    const double seconds = std::chrono::duration<double>(end - start).count();

    std::printf("Result: %.12f\n", result);
    std::printf("Execution Time: %.6f seconds\n", seconds);
    return 0;
}
