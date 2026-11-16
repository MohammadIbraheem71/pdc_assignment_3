// Part 3 light timing sanity check (full benchmarking is Part 4).
//
// Usage: ./timing_sanity [size] [runs] [scene]
//   size   square image side, default 512 (1..4096)
//   runs   timed runs after one warm-up, default 7 (>= 5 required)
//   scene  dense, sparse, overlap or clustered, default clustered
//
// Measures the serial reference and render_parallel at 1, 2, 4 and 8 threads,
// one warm-up plus `runs` timed runs, median with omp_get_wtime(), no I/O in the
// timed regions. T1 is render_parallel on one thread; the serial reference is
// reported separately.
#include "renderer.h"
#include "scenes.h"

#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

volatile float sink = 0.0f;

template <class F>
double median_ms(F&& fn, int runs) {
    fn();  // warm-up, not timed
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(runs));
    for (int i = 0; i < runs; ++i) {
        const double start = omp_get_wtime();
        fn();
        const double stop = omp_get_wtime();
        samples.push_back((stop - start) * 1000.0);
    }
    std::sort(samples.begin(), samples.end());
    const std::size_t mid = samples.size() / 2;
    return samples.size() % 2 ? samples[mid] : 0.5 * (samples[mid - 1] + samples[mid]);
}

}  // namespace

int main(int argc, char** argv) {
    const int size = argc > 1 ? std::atoi(argv[1]) : 512;
    const int runs = argc > 2 ? std::atoi(argv[2]) : 7;
    const std::string scene = argc > 3 ? argv[3] : "clustered";
    if (size <= 0 || size > 4096 || runs < 5) {
        std::fprintf(stderr, "usage: ./timing_sanity [size 1..4096] [runs >= 5] [scene]\n");
        return 2;
    }

    std::vector<Circle> circles;
    if (!make_scene(scene, circles)) {
        std::fprintf(stderr, "unknown scene '%s'\n", scene.c_str());
        return 2;
    }

    omp_set_dynamic(0);
    std::printf("# part 3 timing sanity check\n");
    std::printf("# compiler %s, _OPENMP %d, processors %d\n", __VERSION__, _OPENMP,
                omp_get_num_procs());
    std::printf("# scene %s, %zu circles, %dx%d, %d timed runs after one warm-up\n",
                scene.c_str(), circles.size(), size, size, runs);
    std::printf("# median of omp_get_wtime samples, no I/O in timed regions\n");
    std::printf("# T1 = render_parallel on 1 thread; speedup = T1/Tp; "
                "vs_serial = serial/median\n");

    const Image reference = render_reference(size, size, circles);
    const double serial_ms = median_ms(
        [&] {
            const Image img = render_reference(size, size, circles);
            sink += img.pixels.empty() ? 0.0f : img.pixels[0].r;
        },
        runs);

    std::printf("\n%-10s %7s %12s %9s %10s\n", "variant", "threads", "median_ms",
                "speedup", "vs_serial");
    std::printf("%-10s %7d %12.4f %9s %10.3f\n", "serial", 1, serial_ms, "n/a", 1.0);

    double t1_ms = 0.0;
    for (int p : {1, 2, 4, 8}) {
        if (p > omp_get_num_procs()) {
            std::printf("# note: %d threads skipped, only %d processors available\n", p,
                        omp_get_num_procs());
            continue;
        }
        omp_set_num_threads(p);
        if (!images_close(reference, render_parallel(size, size, circles))) {
            std::printf("# correctness FAIL at %d threads, timing skipped\n", p);
            return 1;
        }
        const double ms = median_ms(
            [&] {
                const Image img = render_parallel(size, size, circles);
                sink += img.pixels.empty() ? 0.0f : img.pixels[0].r;
            },
            runs);
        if (p == 1) t1_ms = ms;
        std::printf("%-10s %7d %12.4f %9.3f %10.3f\n", "parallel", p, ms,
                    p == 1 ? 1.0 : t1_ms / ms, serial_ms / ms);
    }
    (void)sink;
    return 0;
}
