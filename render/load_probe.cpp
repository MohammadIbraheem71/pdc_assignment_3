// Part 3 diagnostic: per-thread work on a clustered scene.
//
// Usage: ./load_probe [size] [scene] [threads]
//   size     square image side, default 1024
//   scene    dense, sparse, overlap or clustered, default clustered
//   threads  worker count, default 8
//
// This probe replicates the baseline's static contiguous row split manually,
// so each thread handles exactly height/threads rows (when divisible). It counts
// covered pixels and blend operations per thread and times each thread's own
// loop with omp_get_wtime after a barrier, which shows how uneven the blend work
// is even though every thread tests the same number of pixels. It is a
// diagnostic, not part of the graded renderer.
#include "renderer.h"
#include "scenes.h"

#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

bool covers(const Circle& c, int x, int y, int w, int h) {
    float px = (x + 0.5f) / w, py = (y + 0.5f) / h;
    float dx = px - c.x, dy = py - c.y;
    return dx * dx + dy * dy <= c.r * c.r;
}

void blend(Pixel& p, const Circle& c) {
    p.r = c.alpha * c.red + (1 - c.alpha) * p.r;
    p.g = c.alpha * c.green + (1 - c.alpha) * p.g;
    p.b = c.alpha * c.blue + (1 - c.alpha) * p.b;
}

}  // namespace

int main(int argc, char** argv) {
    const int size = argc > 1 ? std::atoi(argv[1]) : 1024;
    const std::string scene = argc > 2 ? argv[2] : "clustered";
    const int requested = argc > 3 ? std::atoi(argv[3]) : 8;
    if (size <= 0 || requested < 1) {
        std::fprintf(stderr, "usage: ./load_probe [size] [scene] [threads]\n");
        return 2;
    }

    std::vector<Circle> circles;
    if (!make_scene(scene, circles)) {
        std::fprintf(stderr, "unknown scene '%s'\n", scene.c_str());
        return 2;
    }

    const int threads = requested < omp_get_num_procs() ? requested : omp_get_num_procs();
    std::printf("# per-thread load probe, scene %s, %zu circles, %dx%d, %d threads\n",
                scene.c_str(), circles.size(), size, size, threads);
    std::printf("# static contiguous rows, barrier before timing each thread's loop\n");
    std::printf("%-7s %-14s %-9s %-11s %-13s %-10s\n", "thread", "rows", "pixels",
                "covered", "blends", "loop_ms");

    std::vector<int> rows(static_cast<std::size_t>(threads), 0);
    std::vector<long long> pixels(static_cast<std::size_t>(threads), 0);
    std::vector<long long> covered(static_cast<std::size_t>(threads), 0);
    std::vector<long long> blends(static_cast<std::size_t>(threads), 0);
    std::vector<double> loop_ms(static_cast<std::size_t>(threads), 0.0);

    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        const int begin = tid * size / threads;
        const int end = (tid + 1) * size / threads;

        #pragma omp barrier
        const double start = omp_get_wtime();
        long long local_covered = 0, local_blends = 0;
        for (int y = begin; y < end; ++y) {
            for (int x = 0; x < size; ++x) {
                Pixel p{1, 1, 1};
                bool any = false;
                for (const Circle& c : circles) {
                    if (covers(c, x, y, size, size)) {
                        blend(p, c);
                        ++local_blends;
                        any = true;
                    }
                }
                if (any) ++local_covered;
            }
        }
        const double elapsed = (omp_get_wtime() - start) * 1000.0;

        rows[static_cast<std::size_t>(tid)] = end - begin;
        pixels[static_cast<std::size_t>(tid)] = static_cast<long long>(end - begin) * size;
        covered[static_cast<std::size_t>(tid)] = local_covered;
        blends[static_cast<std::size_t>(tid)] = local_blends;
        loop_ms[static_cast<std::size_t>(tid)] = elapsed;
    }

    long long total_pixels = 0, total_covered = 0, total_blends = 0;
    double min_ms = loop_ms[0], max_ms = loop_ms[0];
    for (int t = 0; t < threads; ++t) {
        std::printf("%-7d %-14d %-9lld %-11lld %-13lld %-10.3f\n", t, rows[t],
                    pixels[t], covered[t], blends[t], loop_ms[t]);
        total_pixels += pixels[t];
        total_covered += covered[t];
        total_blends += blends[t];
        if (loop_ms[t] < min_ms) min_ms = loop_ms[t];
        if (loop_ms[t] > max_ms) max_ms = loop_ms[t];
    }
    std::printf("\npixels=%lld covered=%lld (%.1f%%) blends=%lld\n", total_pixels,
                total_covered,
                100.0 * static_cast<double>(total_covered) / static_cast<double>(total_pixels),
                total_blends);
    std::printf("thread loop_ms min=%.3f max=%.3f ratio=%.2f\n", min_ms, max_ms,
                min_ms > 0.0 ? max_ms / min_ms : 0.0);
    return 0;
}
