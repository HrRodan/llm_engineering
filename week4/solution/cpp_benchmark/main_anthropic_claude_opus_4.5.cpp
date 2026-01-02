
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <vector>

int main() {
    const long long iterations = 200000000LL;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<double> partial_sums(num_threads, 0.0);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    const long long chunk = iterations / num_threads;
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        long long start_i = t * chunk + 1;
        long long end_i = (t == num_threads - 1) ? iterations : (t + 1) * chunk;
        
        threads.emplace_back([start_i, end_i, &partial_sums, t]() {
            double local_sum = 0.0;
            for (long long i = start_i; i <= end_i; ++i) {
                double base = i * 4.0;
                local_sum -= 1.0 / (base - 1.0);
                local_sum += 1.0 / (base + 1.0);
            }
            partial_sums[t] = local_sum;
        });
    }
    
    for (auto& th : threads) {
        th.join();
    }
    
    double result = 1.0;
    for (unsigned int t = 0; t < num_threads; ++t) {
        result += partial_sums[t];
    }
    result *= 4.0;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << std::fixed << std::setprecision(12) << "Result: " << result << std::endl;
    std::cout << std::fixed << std::setprecision(6) << "Execution Time: " << elapsed.count() << " seconds" << std::endl;
    
    return 0;
}
