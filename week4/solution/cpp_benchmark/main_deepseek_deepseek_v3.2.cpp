
#include <iostream>
#include <chrono>
#include <iomanip>

static inline double calculate(int iterations, double param1, double param2) {
    double result = 1.0;
    
    // Unrolled loop for better performance
    int i = 1;
    const int unroll = 4;
    const int limit = iterations - (iterations % unroll);
    
    // Main unrolled loop
    for (; i <= limit; i += unroll) {
        double denom1 = i * param1 - param2;
        double denom2 = i * param1 + param2;
        result -= 1.0 / denom1;
        result += 1.0 / denom2;
        
        double denom3 = (i + 1) * param1 - param2;
        double denom4 = (i + 1) * param1 + param2;
        result -= 1.0 / denom3;
        result += 1.0 / denom4;
        
        double denom5 = (i + 2) * param1 - param2;
        double denom6 = (i + 2) * param1 + param2;
        result -= 1.0 / denom5;
        result += 1.0 / denom6;
        
        double denom7 = (i + 3) * param1 - param2;
        double denom8 = (i + 3) * param1 + param2;
        result -= 1.0 / denom7;
        result += 1.0 / denom8;
    }
    
    // Remainder loop
    for (; i <= iterations; ++i) {
        double denom1 = i * param1 - param2;
        double denom2 = i * param1 + param2;
        result -= 1.0 / denom1;
        result += 1.0 / denom2;
    }
    
    return result;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    double result = calculate(200000000, 4.0, 1.0) * 4.0;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Result: " << result << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Execution Time: " << elapsed.count() << " seconds" << std::endl;
    
    return 0;
}
