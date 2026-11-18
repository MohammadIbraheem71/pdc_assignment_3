// Part 4 schedule and tile-size sweep, plus per-thread busy-time diagnostics.
//
// Usage:
//   ./sweep_part4 [size] [runs] [threads] [scenes]
//   defaults: 1024 5 8 p4_corner,p4_heavy
//
// Sweeps tile sizes 8/16/32/64 with static, dynamic chunks 1/4/16 and guided
// chunk 4 at the requested (largest available) thread count. Every configuration
// is compared with the reference before timing. At tile 16 the static and
// dynamic,4 configurations are re-run with per-thread timers to show imbalance.
//
// Writes part4_sweep.csv and part4_sweep_busy.csv after all timings.
#include "part4_scenes.h"
#include "renderer.h"

#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

volatile float sink = 0.0f;

struct Samples {
    double median = 0.0, minimum = 0.0, maximum = 0.0;
};

Samples summarize(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    Samples result;
    result.median = values.size() % 2 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
    result.minimum = values.front();
    result.maximum = values.back();
    return result;
}

void consume(const Image& image) {
    if (!image.pixels.empty()) sink += image.pixels[0].r;
}

struct Config {
    RenderSchedule schedule;
    int chunk;
    const char* name;
};

struct SweepRow {
    std::string scene;
    int nc = 0, size = 0, tile = 0, chunk = 0, threads = 0;
    std::string schedule;
    Samples total, preprocess, render;
    long long tile_count = 0, candidates = 0;
    double c_avg = 0.0;
};

struct BusyRow {
    std::string scene, schedule;
    int tile = 0, chunk = 0, threads = 0, thread = 0;
    double busy = 0.0, span = 0.0;
};

struct Measurement {
    Samples total, preprocess, render;
    RenderStats info;
};

Measurement measure(int size, const std::vector<Circle>& circles, int p, int runs, int tile,
                    RenderSchedule schedule, int chunk) {
    omp_set_num_threads(p);
    std::vector<double> total, preprocess, render;
    RenderStats last;
    auto once = [&](bool record) {
        RenderStats stats;
        const double start = omp_get_wtime();
        const Image image = render_parallel_optimized(size, size, circles, tile, schedule,
                                                      chunk, &stats);
        const double stop = omp_get_wtime();
        consume(image);
        if (record) {
            total.push_back((stop - start) * 1000.0);
            preprocess.push_back(stats.preprocess_ms);
            render.push_back(stats.render_ms);
            last = stats;
        }
    };
    once(false);
    for (int i = 0; i < runs; ++i) once(true);
    return {summarize(total), summarize(preprocess), summarize(render), last};
}

void print_thread_busy(const std::string& scene, int size, const std::vector<Circle>& circles,
                       int tile, const Config& config, int p, const Image& reference,
                       std::vector<BusyRow>& rows) {
    omp_set_num_threads(p);
    RenderStats busy;
    busy.collect_thread_times = true;
    const Image image = render_parallel_optimized(size, size, circles, tile,
                                                  config.schedule, config.chunk, &busy);
    if (!images_close(reference, image)) {
        std::printf("# %s %s busy diagnostic FAILED correctness\n", scene.c_str(),
                    config.name);
        return;
    }
    double sum_busy = 0.0, max_busy = 0.0, sum_span = 0.0, max_span = 0.0;
    for (int t = 0; t < p; ++t) {
        const double b = busy.thread_busy_ms[static_cast<std::size_t>(t)];
        const double s = busy.thread_span_ms[static_cast<std::size_t>(t)];
        sum_busy += b;
        sum_span += s;
        max_busy = std::max(max_busy, b);
        max_span = std::max(max_span, s);
    }
    const double mean_busy = sum_busy / p, mean_span = sum_span / p;
    std::printf("# busy %s tile=%d %s p=%d: busy mean=%.3f max=%.3f max/mean=%.2f | "
                "span mean=%.3f max=%.3f max/mean=%.2f | idle=%.1f%%\n",
                scene.c_str(), tile, config.name, p, mean_busy, max_busy,
                mean_busy > 0 ? max_busy / mean_busy : 0.0, mean_span, max_span,
                mean_span > 0 ? max_span / mean_span : 0.0,
                mean_span > 0 ? 100.0 * (1.0 - mean_busy / mean_span) : 0.0);
    for (int t = 0; t < p; ++t) {
        const double b = busy.thread_busy_ms[static_cast<std::size_t>(t)];
        const double s = busy.thread_span_ms[static_cast<std::size_t>(t)];
        std::printf("#   thread %d busy_ms=%.3f span_ms=%.3f\n", t, b, s);
        rows.push_back(BusyRow{scene, config.name, tile, config.chunk, p, t, b, s});
    }
}

}  // namespace

int main(int argc, char** argv) {
    const int size = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int runs = argc > 2 ? std::atoi(argv[2]) : 5;
    int p = argc > 3 ? std::atoi(argv[3]) : 8;
    std::vector<std::string> scenes;
    if (argc > 4) {
        std::stringstream in(argv[4]);
        std::string item;
        while (std::getline(in, item, ','))
            if (!item.empty()) scenes.push_back(item);
    } else {
        scenes = {"p4_corner", "p4_heavy"};
    }
    if (size < 1 || size > 4096 || runs < 1 || p < 1 || scenes.empty()) {
        std::fprintf(stderr, "usage: ./sweep_part4 [size 1..4096] [runs >=1] "
                             "[threads >=1] [scene[,scene...]]\n");
        return 2;
    }
    if (p > omp_get_num_procs()) {
        std::printf("# note: %d threads requested, only %d processors available; using %d\n",
                    p, omp_get_num_procs(), omp_get_num_procs());
        p = omp_get_num_procs();
    }

    const Config configs[] = {
        {RenderSchedule::Static, 0, "static"},
        {RenderSchedule::Dynamic, 1, "dynamic,1"},
        {RenderSchedule::Dynamic, 4, "dynamic,4"},
        {RenderSchedule::Dynamic, 16, "dynamic,16"},
        {RenderSchedule::Guided, 4, "guided,4"},
    };
    const int tiles[] = {8, 16, 32, 64};

    std::printf("# Part 4 schedule/tile sweep\n");
    std::printf("# compiler %s, OpenMP macro %d, processors %d\n", __VERSION__, _OPENMP,
                omp_get_num_procs());
    std::printf("# size %dx%d, %d timed runs after one warm-up, %d threads, median ms\n",
                size, size, runs, p);
    std::printf("# every configuration is compared with the reference before timing\n");

    std::vector<SweepRow> rows;
    std::vector<BusyRow> busy_rows;
    int status = 0;

    for (const std::string& scene : scenes) {
        std::vector<Circle> circles;
        if (!make_any_scene(scene, circles)) {
            std::fprintf(stderr, "unknown scene '%s'\n", scene.c_str());
            return 2;
        }
        const Image reference = render_reference(size, size, circles);
        std::printf("\n## scene %s: %zu circles, %dx%d\n", scene.c_str(), circles.size(), size,
                    size);
        std::printf("%-5s %-11s %6s %12s %12s %12s %11s %11s %8s %8s\n", "tile", "schedule",
                    "chunk", "median_ms", "min_ms", "max_ms", "pre_ms", "render_ms",
                    "tiles", "c_avg");

        double best_ms = 1e300;
        std::string best_config;
        for (int tile : tiles) {
            for (const Config& config : configs) {
                omp_set_num_threads(p);
                if (!images_close(reference, render_parallel_optimized(
                                                  size, size, circles, tile,
                                                  config.schedule, config.chunk, nullptr))) {
                    std::printf("# %s tile=%d %s FAILED correctness, skipped\n",
                                scene.c_str(), tile, config.name);
                    status = 1;
                    continue;
                }
                const Measurement m = measure(size, circles, p, runs, tile, config.schedule,
                                              config.chunk);
                const double c_avg = m.info.tile_count > 0
                                         ? static_cast<double>(m.info.candidate_entries) /
                                               m.info.tile_count
                                         : 0.0;
                std::printf("%-5d %-11s %6d %12.4f %12.4f %12.4f %11.4f %11.4f %8lld "
                            "%8.3f\n",
                            tile, config.name, config.chunk, m.total.median, m.total.minimum,
                            m.total.maximum, m.preprocess.median, m.render.median,
                            m.info.tile_count, c_avg);
                if (m.total.median < best_ms) {
                    best_ms = m.total.median;
                    best_config = std::string("tile=") + std::to_string(tile) + " " +
                                  config.name;
                }
                SweepRow row;
                row.scene = scene;
                row.nc = static_cast<int>(circles.size());
                row.size = size;
                row.tile = tile;
                row.chunk = config.chunk;
                row.threads = p;
                row.schedule = config.name;
                row.total = m.total;
                row.preprocess = m.preprocess;
                row.render = m.render;
                row.tile_count = m.info.tile_count;
                row.candidates = m.info.candidate_entries;
                row.c_avg = c_avg;
                rows.push_back(row);
            }
        }
        std::printf("# best %s: %s at %.4f ms\n", scene.c_str(), best_config.c_str(),
                    best_ms);

        // Busy-time diagnostics at tile 16 for static and dynamic chunk 4.
        print_thread_busy(scene, size, circles, 16, configs[0], p, reference, busy_rows);
        print_thread_busy(scene, size, circles, 16, configs[2], p, reference, busy_rows);
    }

    std::ofstream csv("part4_sweep.csv");
    csv << "scene,nc,size,tile,schedule,chunk,threads,median_ms,min_ms,max_ms,"
           "preprocess_ms,render_ms,tile_count,candidate_entries,c_avg\n";
    csv << std::fixed << std::setprecision(6);
    for (const SweepRow& row : rows)
        csv << row.scene << ',' << row.nc << ',' << row.size << ',' << row.tile << ','
            << row.schedule << ',' << row.chunk << ',' << row.threads << ','
            << row.total.median << ',' << row.total.minimum << ',' << row.total.maximum
            << ',' << row.preprocess.median << ',' << row.render.median << ','
            << row.tile_count << ',' << row.candidates << ',' << row.c_avg << '\n';

    std::ofstream busy_csv("part4_sweep_busy.csv");
    busy_csv << "scene,tile,schedule,chunk,threads,thread,busy_ms,span_ms\n";
    busy_csv << std::fixed << std::setprecision(6);
    for (const BusyRow& row : busy_rows)
        busy_csv << row.scene << ',' << row.tile << ',' << row.schedule << ',' << row.chunk
                 << ',' << row.threads << ',' << row.thread << ',' << row.busy << ','
                 << row.span << '\n';

    std::printf("\nwrote part4_sweep.csv (%zu rows) and part4_sweep_busy.csv (%zu rows)\n",
                rows.size(), busy_rows.size());
    return status;
}
