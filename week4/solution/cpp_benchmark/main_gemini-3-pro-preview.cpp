
#include <iostream>
#include <vector>
#include <thread>
#include <iomanip>
#include <chrono>
#include <immintrin.h>

// Worker function for each thread
// Uses AVX2 to process 4 doubles at a time, with 4x loop unrolling (16 elements per iteration)
// This leverages Instruction Level Parallelism to saturate the division units.
void calculate_chunk(long long start, long long end, double p1, double p2, double* result_ptr) {
    double sum = 0.0;
    
    // Algebraic optimization:
    // The Python code calculates: -1/(p1*i - p2) + 1/(p1*i + p2)
    // Combined term: ((p1*i - p2) - (p1*i + p2)) / ((p1*i - p2)(p1*i + p2))
    //              = -2*p2 / ((p1*i)^2 - p2^2)
    // For p1=4, p2=1: -2 / (16*i^2 - 1)
    // This reduces the number of divisions from 2 to 1 per iteration.
    
    double num_scalar = -2.0 * p2;
    double c1_scalar = p1 * p1;
    double c2_scalar = p2 * p2;

    long long i = start;
    
    // AVX2 Vectorization
    // We compute -2.0 / (c1 * i^2 - c2)
    
    // Process 16 elements per loop iteration
    long long end_vec = end - 15;

    if (i <= end_vec) {
        __m256d v_c1 = _mm256_set1_pd(c1_scalar);
        __m256d v_c2 = _mm256_set1_pd(c2_scalar);
        __m256d v_num = _mm256_set1_pd(num_scalar);
        
        // Accumulators for unrolled loop
        __m256d v_sum0 = _mm256_setzero_pd();
        __m256d v_sum1 = _mm256_setzero_pd();
        __m256d v_sum2 = _mm256_setzero_pd();
        __m256d v_sum3 = _mm256_setzero_pd();

        // Index vectors: i, i+1, i+2, ...
        __m256d v_i0 = _mm256_setr_pd((double)i, (double)(i+1), (double)(i+2), (double)(i+3));
        __m256d v_i1 = _mm256_setr_pd((double)(i+4), (double)(i+5), (double)(i+6), (double)(i+7));
        __m256d v_i2 = _mm256_setr_pd((double)(i+8), (double)(i+9), (double)(i+10), (double)(i+11));
        __m256d v_i3 = _mm256_setr_pd((double)(i+12), (double)(i+13), (double)(i+14), (double)(i+15));
        
        __m256d v_inc = _mm256_set1_pd(16.0);

        for (; i <= end_vec; i += 16) {
            // i^2
            __m256d i2_0 = _mm256_mul_pd(v_i0, v_i0);
            __m256d i2_1 = _mm256_mul_pd(v_i1, v_i1);
            __m256d i2_2 = _mm256_mul_pd(v_i2, v_i2);
            __m256d i2_3 = _mm256_mul_pd(v_i3, v_i3);

            // Denominator: c1*i^2 - c2
            // Uses Fused Multiply-Subtract
            __m256d den0 = _mm256_fmsub_pd(v_c1, i2_0, v_c2);
            __m256d den1 = _mm256_fmsub_pd(v_c1, i2_1, v_c2);
            __m256d den2 = _mm256_fmsub_pd(v_c1, i2_2, v_c2);
            __m256d den3 = _mm256_fmsub_pd(v_c1, i2_3, v_c2);

            // Add num / den
            v_sum0 = _mm256_add_pd(v_sum0, _mm256_div_pd(v_num, den0));
            v_sum1 = _mm256_add_pd(v_sum1, _mm256_div_pd(v_num, den1));
            v_sum2 = _mm256_add_pd(v_sum2, _mm256_div_pd(v_num, den2));
            v_sum3 = _mm256_add_pd(v_sum3, _mm256_div_pd(v_num, den3));

            // Increment indices
            v_i0 = _mm256_add_pd(v_i0, v_inc);
            v_i1 = _mm256_add_pd(v_i1, v_inc);
            v_i2 = _mm256_add_pd(v_i2, v_inc);
            v_i3 = _mm256_add_pd(v_i3, v_inc);
        }
        
        // Reduce vectors to scalar
        __m256d v_total = _mm256_add_pd(_mm256_add_pd(v_sum0, v_sum1), _mm256_add_pd(v_sum2, v_sum3));
        double tmp[4];
        _mm256_storeu_pd(tmp, v_total);
        sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }

    // Handle remaining iterations with original scalar logic to ensure boundary precision
    for (; i <= end; ++i) {
        double j1 = i * p1 - p2;
        double j2 = i * p1 + p2;
        sum -= (1.0 / j1);
        sum += (1.0 / j2);
    }

    *result_ptr = sum;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    const int iterations = 200000000;
    const double param1 = 4.0;
    const double param2 = 1.0;

    // Detect core count for multithreading
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    // Align results to avoid false sharing could be an optimization, but vector alloc usually spaces out well enough for this simple case.
    // Given the compute intensity vs memory access, false sharing is negligible here.
    std::vector<double> results(num_threads);

    long long chunk_size = iterations / num_threads;
    long long start = 1;

    for (unsigned int t = 0; t < num_threads; ++t) {
        long long end = (t == num_threads - 1) ? iterations : (start + chunk_size - 1);
        threads.emplace_back(calculate_chunk, start, end, param1, param2, &results[t]);
        start = end + 1;
    }

    double total_sum = 1.0;
    for (unsigned int t = 0; t < num_threads; ++t) {
        threads[t].join();
        total_sum += results[t];
    }
    
    double final_result = total_sum * 4.0;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Result: " << final_result << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Execution Time: " << diff.count() << " seconds" << std::endl;

    return 0;
}
