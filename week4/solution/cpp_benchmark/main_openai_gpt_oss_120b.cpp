#include <cstdio>
#include <chrono>

int main() {
    const int iterations = 200000000;
    const double param1 = 4.0;
    const double param2 = 1.0;
    double result = 1.0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 1; i <= iterations; ++i) {
        double j = i * param1 - param2;
        result -= 1.0 / j;
        j = i * param1 + param2;
        result += 1.0 / j;
    }

    result *= 4.0;

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::printf("Result: %.12f\n", result);
    std::printf("Execution Time: %.6f seconds\n", elapsed);
    return 0;
}
