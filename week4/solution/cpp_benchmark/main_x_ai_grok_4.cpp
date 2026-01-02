
#include <iostream>
#include <iomanip>
#include <chrono>

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    double result = 1.0;
    double denom = 3.0;
    const long long iterations = 200000000LL;
    for (long long i = 0; i < iterations; ++i) {
        result -= 1.0 / denom;
        denom += 2.0;
        result += 1.0 / denom;
        denom += 2.0;
    }
    result *= 4.0;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Result: " << std::fixed << std::setprecision(12) << result << std::endl;
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) << duration.count() << " seconds" << std::endl;

    return 0;
}
