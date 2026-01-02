
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <immintrin.h>
#include <chrono>
#include <cstdio>

// Constants
const int ITERATIONS = 200000000;
const double PARAM1 = 4.0;
const double PARAM2 = 1.0;
const int BLOCK_SIZE = 100000; 

// Atomic counter for dynamic load balancing
std::atomic<int> next_block(1);

// Structure to align partial results and avoid false sharing
struct alignas(64) AlignedResult {
    double value;
};
std::vector<AlignedResult> partial_results;

void worker(int thread_id) {
    double local_sum = 0.0;
    
    // AVX2 constants
    __m256d v_p1 = _mm256_set1_pd(PARAM1);
    __m256d v_p2 = _mm256_set1_pd(PARAM2);
    __m256d v_one = _mm256_set1_pd(1.0);
    __m256d v_res = _mm256_setzero_pd();
    __m256d v_four = _mm256_set1_pd(4.0);

    while (true) {
        int start = next_block.fetch_add(BLOCK_SIZE, std::memory_order_relaxed);
        if (start > ITERATIONS) break;
        
        int end = start + BLOCK_SIZE;
        if (end > ITERATIONS + 1) end = ITERATIONS + 1;
        
        int i = start;
        int limit = end - 3;
        
        // Initialize vector i: [i+3, i+2, i+1, i]
        __m256d v_i = _mm256_set_pd(
            (double)(i + 3),
            (double)(i + 2),
            (double)(i + 1),
            (double)i
        );
        
        // Main vectorized loop
        for (; i < limit; i += 4) {
            // j1 = i * 4 - 1
            __m256d v_j1 = _mm256_fmsub_pd(v_i, v_p1, v_p2);
            // res -= 1/j1
            v_res = _mm256_sub_pd(v_res, _mm256_div_pd(v_one, v_j1));
            
            // j2 = i * 4 + 1
            __m256d v_j2 = _mm256_fmadd_pd(v_i, v_p1, v_p2);
            // res += 1/j2
            v_res = _mm256_add_pd(v_res, _mm256_div_pd(v_one, v_j2));
            
            // Increment index vector
            v_i = _mm256_add_pd(v_i, v_four);
        }
        
        // Scalar cleanup for remaining iterations in block
        for (; i < end; ++i) {
            double j1 = i * PARAM1 - PARAM2;
            local_sum -= (1.0 / j1);
            double j2 = i * PARAM1 + PARAM2;
            local_sum += (1.0 / j2);
        }
    }
    
    // Horizontal reduction of vector accumulator
    double buffer[4];
    _mm256_storeu_pd(buffer, v_res);
    local_sum += buffer[0] + buffer[1] + buffer[2] + buffer[3];
    
    partial_results[thread_id].value = local_sum;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    std::vector<std::thread> threads;
    partial_results.resize(num_threads);

    for (unsigned int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }

    for (auto& t : threads) {
        t.join();
    }

    double total_sum = 1.0; 
    for (const auto& res : partial_results) {
        total_sum += res.value;
    }
    
    double final_result = total_sum * 4.0;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    printf("Result: %.12f\n", final_result);
    printf("Execution Time: %.6f seconds\n", duration.count());

    return 0;
}
