
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdint>

double calculate(int64_t iterations, int param1, int param2) {
    double result = 1.0;
    
    // Optimized loop: compiler will unroll and vectorize with -O3 -march=native
    // Using 64-bit integer for loop counter to match large iteration count size
    for (int64_t i = 1; i <= iterations; ++i) {
        // Replicate Python logic exactly to ensure floating point consistency
        // 1.0 / integer is promoted to double division
        result -= 1.0 / (i * param1 - param2);
        result += 1.0 / (i * param1 + param2);
    }
    return result;
}

int main() {
    using namespace std::chrono;

    // High-resolution timer start
    auto start_time = high_resolution_clock::now();

    // Constants defined exactly as in Python
    const int64_t iterations = 200000000;
    const int param1 = 4;
    const int param2 = 1;

    // Execute calculation
    double result = calculate(iterations, param1, param2) * 4;

    // High-resolution timer end
    auto end_time = high_resolution_clock::now();
    
    // Calculate duration in seconds
    duration<double> elapsed = end_time - start_time;

    // Format and print output
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Result: " << result << "\n";
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Execution Time: " << elapsed.count() << " seconds\n";

    return 0;
}
