
/* 
 * Optimization Explanation:
 * 1. Multithreading: The code identifies the number of hardware threads (20 on i7-13700H) and 
 *    splits the 200,000,000 iterations into even chunks. This utilizes all CPU cores, 
 *    drastically reducing wall-clock time compared to Python's single-threaded execution.
 * 2. SIMD (AVX2): Inside each thread, the loop is implemented using AVX2 intrinsics (__m256d).
 *    This processes 4 double-precision floating-point numbers simultaneously per instruction cycle.
 * 3. Vectorized Division: Floating point division is expensive. AVX2 allows performing 4 divisions 
 *    in parallel (`_mm256_div_pd`), improving throughput.
 * 4. Index Management: Indices are maintained in a vector register and incremented by 4s to 
 *    avoid scalar-to-vector transfer overhead inside the hot loop.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <future>
#include <chrono>
#include <immintrin.h>

// Worker function optimized with AVX2 intrinsics to process a range of iterations
double calculate_segment(long long start, long long end, double param1, double param2) {
    double sum = 0.0;
    
    // Broadcast constants to AVX registers
    __m256d p1_vec = _mm256_set1_pd(param1);
    __m256d p2_vec = _mm256_set1_pd(param2);
    __m256d one_vec = _mm256_set1_pd(1.0);
    __m256d accum = _mm256_setzero_pd();
    __m256d step_vec = _mm256_set1_pd(4.0);
    
    // Initialize indices vector: [start, start+1, start+2, start+3]
    // _mm256_set_pd expects inputs in reverse order (e3, e2, e1, e0)
    __m256d indices = _mm256_set_pd((double)(start + 3), (double)(start + 2), (double)(start + 1), (double)start);

    long long i = start;
    long long limit = end - 3;

    // Main vectorized loop
    for (; i <= limit; i += 4) {
        // Calculate denominators parallelly
        // j1 = indices * param1 - param2
        __m256d j1 = _mm256_sub_pd(_mm256_mul_pd(indices, p1_vec), p2_vec);
        
        // j2 = indices * param1 + param2
        __m256d j2 = _mm256_add_pd(_mm256_mul_pd(indices, p1_vec), p2_vec);
        
        // Calculate reciprocals parallelly
        // term1 = 1.0 / j1
        __m256d term1 = _mm256_div_pd(one_vec, j1);
        // term2 = 1.0 / j2
        __m256d term2 = _mm256_div_pd(one_vec, j2);
        
        // Accumulate: result += term2 - term1
        accum = _mm256_add_pd(accum, _mm256_sub_pd(term2, term1));
        
        // Increment indices by 4 for next iteration
        indices = _mm256_add_pd(indices, step_vec);
    }

    // Horizontal sum of the SIMD accumulator
    double temp[4];
    _mm256_storeu_pd(temp, accum);
    sum += temp[0] + temp[1] + temp[2] + temp[3];

    // Handle any remaining iterations (scalar cleanup)
    for (; i <= end; ++i) {
        double j1 = i * param1 - param2;
        double j2 = i * param1 + param2;
        sum += (1.0 / j2) - (1.0 / j1);
    }

    return sum;
}

int main() {
    // Parameters matching the Python script
    const long long iterations = 200000000;
    const double param1 = 4.0;
    const double param2 = 1.0;

    // Start high-resolution timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Determine number of concurrent threads supported by the hardware
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Safe fallback
    
    // Container for futures to retrieve results from threads
    std::vector<std::future<double>> futures;
    futures.reserve(num_threads);
    
    long long chunk_size = iterations / num_threads;
    long long start = 1;
    
    // Launch threads
    for (unsigned int t = 0; t < num_threads; ++t) {
        // Ensure the last thread covers all remaining iterations
        long long end = (t == num_threads - 1) ? iterations : (start + chunk_size - 1);
        
        futures.push_back(std::async(std::launch::async, calculate_segment, start, end, param1, param2));
        start = end + 1;
    }

    // Initialize result with 1.0 as per Python logic
    double total_result = 1.0; 
    
    // Aggregate results from all threads
    for (auto &f : futures) {
        total_result += f.get();
    }
    
    // Apply final multiplication
    total_result *= 4.0;

    // End timer
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    // Print output in identical format to Python script
    std::cout << "Result: " << std::fixed << std::setprecision(12) << total_result << std::endl;
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) << duration.count() << " seconds" << std::endl;

    return 0;
}
