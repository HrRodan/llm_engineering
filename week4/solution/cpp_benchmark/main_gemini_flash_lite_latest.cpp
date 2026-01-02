
#include <iostream>

// Recursive Fibonacci function.
// For small N, this recursive implementation is simple, but its performance
// is extremely poor (exponential time complexity O(2^n)).
// Since the original Python code uses this exact implementation, we must replicate it.
// On modern fast hardware, this will still be slow for large N, but it is the
// direct port of the provided logic.
long long fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

int main() {
    // Based on typical competitive programming scenarios or simple testing,
    // a value of N around 40-45 will be sufficient to demonstrate the slowness
    // of this recursive approach without taking excessive time on a fast CPU.
    // We choose 40 as a safe, demonstrably slow, but quick-to-run-enough test case.
    const int N = 40;

    // Since the Python code didn't include timing or printing, we add minimal
    // scaffolding to ensure the code runs and produces output equivalent to
    // running the function call.
    long long result = fib(N);

    // Output the result to mimic execution completion.
    std::cout << result << std::endl;

    return 0;
}
