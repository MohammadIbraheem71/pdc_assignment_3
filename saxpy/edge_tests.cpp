// Part 1 extra edge-case tests. This is a separate program; it does not modify
// the supplied demo, benchmark, reference implementation, or headers.
//
// Build/run from the repository root with:
//   ./saxpy/run_tests.sh
//
// Cases: n = 0, n = 1, n smaller than the thread count, lengths not divisible
// by the thread count, and a large deterministic random array. Every case is
// checked at 1, 2, 4 and 8 threads (skipping counts the machine cannot run).
#include "saxpy.h"
#include "../common/rng.h"

#include <omp.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

int checks = 0;
int failures = 0;

bool same_values(const std::vector<float>& expected, const std::vector<float>& actual,
                 float tolerance = 1e-4f) {
    if (expected.size() != actual.size()) {
        std::printf("  size mismatch: expected %zu, got %zu\n", expected.size(), actual.size());
        return false;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(expected[i] - actual[i]) > tolerance) {
            std::printf("  first mismatch at index %zu: expected %.9g, got %.9g\n", i,
                        expected[i], actual[i]);
            return false;
        }
    }
    return true;
}

// Runs the serial reference once, then the parallel version at each thread
// count. The input vectors are copied beforehand so we can also verify that the
// implementation does not modify its inputs.
void check_case(const char* name, float a, const std::vector<float>& x,
                const std::vector<float>& y, const std::vector<int>& threads) {
    const std::vector<float> x_before = x;
    const std::vector<float> y_before = y;
    const std::vector<float> expected = saxpy_serial(a, x, y);

    for (int p : threads) {
        omp_set_num_threads(p);
        const std::vector<float> actual = saxpy_parallel(a, x, y);
        ++checks;
        const bool values_ok = same_values(expected, actual);
        const bool inputs_ok = (x == x_before) && (y == y_before);
        const bool ok = values_ok && inputs_ok;
        if (!ok) ++failures;
        std::printf("%-24s n=%-8zu threads=%-2d values=%-4s inputs=%-4s %s\n", name, x.size(), p,
                    values_ok ? "PASS" : "FAIL", inputs_ok ? "PASS" : "FAIL",
                    ok ? "PASS" : "FAIL");
    }
}

void check_size_mismatch() {
    bool threw = false;
    try {
        const std::vector<float> result = saxpy_parallel(2.0f, {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f});
        (void)result;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ++checks;
    if (!threw) ++failures;
    std::printf("%-24s %s\n", "size mismatch throws", threw ? "PASS" : "FAIL");
}

}  // namespace

int main() {
    const int available = omp_get_num_procs();
    std::vector<int> threads;
    for (int p : {1, 2, 4, 8}) {
        if (p <= available)
            threads.push_back(p);
        else
            std::printf("# note: %d threads skipped, only %d processors available\n", p, available);
    }

    std::printf("SAXPY Part 1 edge-case tests\n");
    std::printf("processors available: %d, thread counts tested: ", available);
    for (std::size_t i = 0; i < threads.size(); ++i)
        std::printf("%s%d", i == 0 ? "" : ",", threads[i]);
    std::printf("\n\n");

    check_case("empty", 1.5f, {}, {}, threads);
    check_case("single element", 2.0f, {3.5f}, {-1.25f}, threads);

    // n smaller than the largest thread count, and not divisible by it.
    check_case("smaller than threads", -1.5f, {1, 2, 3}, {4, 5, 6}, threads);
    check_case("odd length 7", 0.5f, {1, 2, 3, 4, 5, 6, 7}, {-1, -2, -3, -4, -5, -6, -7},
               threads);
    check_case("odd length 1001", 1.25f,
               std::vector<float>(1001, 0.75f), std::vector<float>(1001, -0.25f), threads);

    // Large deterministic random array (same generator as scenes/benchmark inputs).
    const std::size_t large_n = 10000000;
    Lcg rng(2024);
    std::vector<float> x(large_n), y(large_n);
    for (std::size_t i = 0; i < large_n; ++i) {
        x[i] = rng.range(-10.0f, 10.0f);
        y[i] = rng.range(-10.0f, 10.0f);
    }
    check_case("large random 1e7", 1.5f, x, y, threads);

    check_size_mismatch();

    std::printf("\nEDGE TESTS: %d checks, %d failed -> %s\n", checks, failures,
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
