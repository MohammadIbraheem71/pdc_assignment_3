// Diagnostic probe (not part of the submitted SAXPY implementation).
// Compares three timed variants at n=1e7:
//   full   : saxpy_parallel as submitted (allocate + zero-fill + parallel loop)
//   inplace: preallocated output, parallel loop only
//   zero   : allocating and zero-filling the output vector only (serial)
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 10000000;
    const int runs = argc > 2 ? std::atoi(argv[2]) : 9;
    const float a = 2.5f;
    std::vector<float> x(n), y(n), z(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = float(i % 97) / 10.0f;
        y[i] = float(i % 17);
    }
    volatile float sink = 0.0f;

    const double zero_ms = median_ms(
        [&] {
            std::vector<float> out(n, 0.0f);
            sink += out[0];
        },
        runs);
    std::printf("n=%zu zero_fill_only_ms=%.4f\n", n, zero_ms);

    for (int p : {1, 2, 4, 8}) {
        omp_set_num_threads(p);
        const double inplace = median_ms(
            [&] {
                #pragma omp parallel for
                for (std::size_t i = 0; i < n; ++i) z[i] = a * x[i] + y[i];
                sink += z[0];
            },
            runs);
        const double full = median_ms(
            [&] {
                std::vector<float> out(n, 0.0f);
                #pragma omp parallel for
                for (std::size_t i = 0; i < n; ++i) out[i] = a * x[i] + y[i];
                sink += out[0];
            },
            runs);
        std::printf("threads=%d inplace_ms=%.4f full_ms=%.4f\n", p, inplace, full);
    }
    (void)sink;
    return 0;
}
