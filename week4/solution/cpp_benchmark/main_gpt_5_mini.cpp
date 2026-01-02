#include <bits/stdc++.h>
using namespace std;
using u64 = unsigned long long;

double calculate(u64 iterations, int param1, int param2) {
    double result = 1.0;
    // optimize by avoiding multiplication inside loop: jbase = i * param1
    double jbase = double(param1); // for i = 1
    const double p2 = double(param2);
    for (u64 i = 1; i <= iterations; ++i) {
        double j = jbase - p2;
        result -= 1.0 / j;
        j = jbase + p2;
        result += 1.0 / j;
        jbase += double(param1);
    }
    return result;
}

int main() {
    const u64 iterations = 200000000ULL;
    const int param1 = 4;
    const int param2 = 1;

    auto t1 = chrono::high_resolution_clock::now();
    double result = calculate(iterations, param1, param2) * 4.0;
    auto t2 = chrono::high_resolution_clock::now();

    double elapsed = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();

    // match Python formatting
    printf("Result: %.12f\n", result);
    printf("Execution Time: %.6f seconds\n", elapsed);

    return 0;
}