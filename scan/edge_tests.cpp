// Part 2 extra tests. Separate from the supplied demo and benchmark.
//
// Build/run from the repository root:
//   ./scan/run_tests.sh
//
// Covers the required lengths, random arrays (fixed seed), and all-equal,
// all-distinct, alternating-distinct and alternating-pair patterns. Every case
// runs at 1, 2, 4 and 8 threads where supported.
#include "scan.h"
#include "../common/rng.h"

#include <omp.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

int checks = 0;
int failures = 0;

bool same_values(const char* op, const std::vector<int>& expected,
                 const std::vector<int>& actual) {
    if (expected.size() != actual.size()) {
        std::printf("  %s size mismatch: expected %zu, got %zu\n", op, expected.size(),
                    actual.size());
        return false;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            std::printf("  %s first mismatch at index %zu: expected %d, got %d\n", op, i,
                        expected[i], actual[i]);
            return false;
        }
    }
    return true;
}

void check_case(const std::string& name, const std::vector<int>& input,
                const std::vector<int>& threads) {
    const std::vector<int> expected_scan = exclusive_scan_serial(input);
    const std::vector<int> expected_repeats = find_repeats_serial(input);
    const std::vector<int> input_before = input;

    for (int p : threads) {
        omp_set_num_threads(p);
        const std::vector<int> scan = exclusive_scan_parallel(input);
        const std::vector<int> repeats = find_repeats_parallel(input);
        const bool scan_ok = same_values("scan", expected_scan, scan);
        const bool repeats_ok = same_values("repeats", expected_repeats, repeats);
        const bool inputs_ok = input == input_before;
        const bool ok = scan_ok && repeats_ok && inputs_ok;
        ++checks;
        if (!ok) ++failures;
        std::printf("%-26s n=%-9zu p=%-2d scan=%-4s repeats=%-4s inputs=%-4s %s\n",
                    name.c_str(), input.size(), p, scan_ok ? "PASS" : "FAIL",
                    repeats_ok ? "PASS" : "FAIL", inputs_ok ? "PASS" : "FAIL",
                    ok ? "PASS" : "FAIL");
    }
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

    std::printf("SCAN Part 2 edge-case tests\n");
    std::printf("processors available: %d, thread counts tested: ", available);
    for (std::size_t i = 0; i < threads.size(); ++i)
        std::printf("%s%d", i == 0 ? "" : ",", threads[i]);
    std::printf("\n\n");

    // Required assignment lengths, same deterministic input as scan_demo.
    for (std::size_t n : {0u, 1u, 7u, 8u, 9u, 1024u, 1025u}) {
        std::vector<int> input(n);
        for (std::size_t i = 0; i < n; ++i) input[i] = static_cast<int>((i / 3) % 7);
        check_case("required length " + std::to_string(n), input, threads);
    }

    // Examples from the header comments.
    check_case("scan example", {1, 4, 6, 8, 2}, threads);
    check_case("repeats example", {1, 2, 2, 1, 1, 1, 3, 5, 3, 3}, threads);

    // Patterned inputs for find_repeats (and the scan).
    check_case("all equal 1025", std::vector<int>(1025, 7), threads);
    check_case("all equal 10000", std::vector<int>(10000, -3), threads);

    std::vector<int> distinct(1025);
    std::vector<int> alternating(1025);
    std::vector<int> pairs(1025);
    for (std::size_t i = 0; i < 1025; ++i) {
        distinct[i] = static_cast<int>(i);
        alternating[i] = static_cast<int>(i % 2);
        pairs[i] = static_cast<int>(i / 2);
    }
    check_case("all distinct 1025", distinct, threads);
    check_case("alternating 0/1 1025", alternating, threads);
    check_case("alternating pairs 1025", pairs, threads);

    // Deterministic pseudo-random arrays, including a large one.
    for (std::size_t n : {2u, 3u, 17u, 1025u, 100000u, 1000000u, 10000000u}) {
        Lcg rng(424242);
        std::vector<int> input(n);
        for (std::size_t i = 0; i < n; ++i) input[i] = rng.next_int(4);
        check_case("random " + std::to_string(n), input, threads);
    }

    std::printf("\nEDGE TESTS: %d checks, %d failed -> %s\n", checks, failures,
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
