// Diagnostic probe 2 (not part of the submitted scan implementation).
// Build/run from the repository root:
//   g++ -O2 -std=c++17 -Wall -Wextra -fopenmp -I scan \
//       scan/probe_scan.cpp scan/scan.cpp -o /tmp/probe_scan
//   /tmp/probe_scan 10000000 9
//
// Decomposes the per-call cost of exclusive_scan_parallel into
//   1. fresh tree allocation + parallel init + result copy (alloc_ms),
//   2. the up/down sweep on a preallocated tree (sweep_ms),
//   3. the submitted function end to end (full_ms).
// A dummy warm-up call is made first so the allocator is in a steady state.
#include "scan.h"

#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace {
volatile long long sink = 0;

template <class F>
double median_ms(F&& fn, int runs) {
    fn();
    std::vector<double> samples;
    for (int i = 0; i < runs; ++i) {
        const double t0 = omp_get_wtime();
        fn();
        const double t1 = omp_get_wtime();
        samples.push_back((t1 - t0) * 1000.0);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void sweep(std::unique_ptr<int[]>& tree, const std::vector<int>& input, std::size_t N) {
    const std::size_t n = input.size();
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < N; ++i) tree[i] = i < n ? input[i] : 0;
        for (std::size_t d = 1; d < N; d <<= 1) {
            #pragma omp for schedule(static)
            for (std::size_t k = 0; k < N; k += 2 * d)
                tree[k + 2 * d - 1] += tree[k + d - 1];
        }
        #pragma omp single
        tree[N - 1] = 0;
        for (std::size_t d = N >> 1; d > 0; d >>= 1) {
            #pragma omp for schedule(static)
            for (std::size_t k = 0; k < N; k += 2 * d) {
                const int left = tree[k + d - 1];
                tree[k + d - 1] = tree[k + 2 * d - 1];
                tree[k + 2 * d - 1] += left;
            }
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 10000000;
    const int runs = argc > 2 ? std::atoi(argv[2]) : 9;
    std::size_t N = 1;
    while (N < n) N <<= 1;

    std::vector<int> input(n);
    for (std::size_t i = 0; i < n; ++i) input[i] = static_cast<int>((i / 3) % 7);
    std::unique_ptr<int[]> tree(new int[N]);

    // Warm the allocator exactly as the first timed call would.
    (void)exclusive_scan_parallel(input);

    for (int p : {1, 2, 4, 8}) {
        omp_set_num_threads(p);

        const double full = median_ms(
            [&] {
                const std::vector<int> result = exclusive_scan_parallel(input);
                sink += result[0];
            },
            runs);

        const double sweep_ms = median_ms(
            [&] {
                sweep(tree, input, N);
                sink += tree[0];
            },
            runs);

        const double alloc_ms = median_ms(
            [&] {
                std::unique_ptr<int[]> fresh(new int[N]);
                #pragma omp parallel for schedule(static)
                for (std::size_t i = 0; i < N; ++i) fresh[i] = i < n ? input[i] : 0;
                const std::vector<int> result(fresh.get(), fresh.get() + n);
                sink += result[0];
            },
            runs);

        std::printf("n=%zu p=%d full_ms=%.4f sweep_ms=%.4f alloc_init_copy_ms=%.4f\n", n, p,
                    full, sweep_ms, alloc_ms);
    }
    (void)sink;
    return 0;
}
