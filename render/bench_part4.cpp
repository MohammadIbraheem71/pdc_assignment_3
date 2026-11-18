// Part 4 main benchmark: baseline vs optimized, medians and speedups.
//
// Usage:
//   ./bench_part4 [size] [runs] [threads] [scenes] [tile] [schedule] [chunk]
//   defaults: 1024 5 1,2,4,8 p4_sparse,p4_heavy,p4_corner,p4_few,p4_one,p4_huge 16 static 0
//
//   schedule is static, dynamic, guided, auto or runtime.
//   chunk <= 0 leaves the schedule's default chunk.
//
// For each scene the reference is rendered first and BOTH renderers are compared
// with it at every thread count. If any comparison fails, all timings for that
// scene are skipped and the program exits with status 1.
//
// Conventions:
//   T1 is the baseline render_parallel() on one thread for that scene and is
//   used for every speedup Sp = T1/Tp. self_speedup is the version's own
//   T1(version)/Tp(version). efficiency = Sp/p.
//   The optimized preprocessing/render split is measured inside the function;
//   the total is measured externally with omp_get_wtime(). No file output is
//   inside a timed region.
//
// Writes part4_results.csv and part4_busy.csv after all measurements.
#include "../common/timing.h"
#include "part4_scenes.h"
#include "renderer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace {

volatile float sink = 0.0f;

struct Samples {
    double median = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
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

struct Row {
    std::string scene, variant, schedule;
    int nc = 0, size = 0, tile = 0, chunk = 0, threads = 0;
    Samples total;
    double preprocess = 0.0, render = 0.0;
    double speedup = 0.0, self_speedup = 0.0, efficiency = 0.0;
    long long tile_count = 0, candidates = 0;
    double c_avg = 0.0, t_avg = 0.0, predicted = 0.0;
};

struct BusyRow {
    std::string scene, schedule;
    int tile = 0, chunk = 0, threads = 0, thread = 0;
    double busy = 0.0, span = 0.0;
};

bool parse_schedule(const std::string& text, RenderSchedule& schedule) {
    if (text == "static") schedule = RenderSchedule::Static;
    else if (text == "dynamic") schedule = RenderSchedule::Dynamic;
    else if (text == "guided") schedule = RenderSchedule::Guided;
    else if (text == "auto") schedule = RenderSchedule::Auto;
    else if (text == "runtime") schedule = RenderSchedule::Runtime;
    else return false;
    return true;
}

Samples time_baseline(int size, const std::vector<Circle>& circles, int p, int runs) {
    omp_set_num_threads(p);
    omp_set_schedule(omp_sched_static, 0);
    auto once = [&] {
        const Image image = render_parallel(size, size, circles);
        consume(image);
    };
    once();  // warm-up
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(runs));
    for (int i = 0; i < runs; ++i) {
        const double start = omp_get_wtime();
        once();
        samples.push_back((omp_get_wtime() - start) * 1000.0);
    }
    return summarize(samples);
}

struct OptimizedResult {
    Samples total, preprocess, render;
    RenderStats info;
};

OptimizedResult time_optimized(int size, const std::vector<Circle>& circles, int p, int runs,
                               int tile, RenderSchedule schedule, int chunk) {
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
    once(false);  // warm-up
    for (int i = 0; i < runs; ++i) once(true);
    return {summarize(total), summarize(preprocess), summarize(render), last};
}

void print_bench_context(int size, int runs, int tile, const std::string& schedule, int chunk) {
    std::printf("# Part 4 benchmark\n");
    std::printf("# compiler %s, OpenMP macro %d, processors %d\n", __VERSION__, _OPENMP,
                omp_get_num_procs());
    std::printf("# size %dx%d, %d timed runs after one warm-up, medians with omp_get_wtime\n",
                size, size, runs);
    std::printf("# baseline schedule static; optimized tile %d, schedule %s", tile,
                schedule.c_str());
    if (chunk > 0 && schedule != "static" && schedule != "runtime")
        std::printf(" chunk %d", chunk);
    std::printf("\n");
    std::printf("# T1 = baseline render_parallel on 1 thread for each scene\n");
    std::printf("# speedup = T1/median, efficiency = speedup/p, self = version T1/version Tp\n");
    std::printf("# optimized preprocess+render are internal medians; total is external\n");
    std::printf("# file output happens only after all timed regions\n");
}

}  // namespace

int main(int argc, char** argv) {
    const int size = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int runs = argc > 2 ? std::atoi(argv[2]) : 5;
    const std::vector<int> threads =
        usable_threads(parse_list(argc > 3 ? argv[3] : "1,2,4,8"));
    const std::vector<std::string> scenes =
        split_names(argc > 4 ? argv[4]
                             : "p4_sparse,p4_heavy,p4_corner,p4_few,p4_one,p4_huge");
    const int tile = argc > 5 ? std::atoi(argv[5]) : 16;
    const std::string schedule_name = argc > 6 ? argv[6] : "static";
    const int chunk = argc > 7 ? std::atoi(argv[7]) : 0;

    RenderSchedule schedule;
    if (size < 1 || size > 4096 || runs < 1 || tile < 1 || scenes.empty() ||
        !parse_schedule(schedule_name, schedule)) {
        std::fprintf(stderr,
                     "usage: ./bench_part4 [size 1..4096] [runs >=1] [threads] [scenes] "
                     "[tile >=1] [static|dynamic|guided|auto|runtime] [chunk]\n");
        return 2;
    }

    print_bench_context(size, runs, tile, schedule_name, chunk);
    omp_set_dynamic(0);

    std::vector<Row> rows;
    std::vector<BusyRow> busy_rows;
    int status = 0;

    for (const std::string& scene : scenes) {
        std::vector<Circle> circles;
        if (!make_any_scene(scene, circles)) {
            std::fprintf(stderr, "unknown scene '%s'\n", scene.c_str());
            return 2;
        }

        const Image reference = render_reference(size, size, circles);
        const double serial_ms = time_once_ms([&] {
            const Image image = render_reference(size, size, circles);
            consume(image);
        });

        // Compare everything before recording any time for this scene.
        bool valid = true;
        RenderStats info;
        for (int p : threads) {
            omp_set_num_threads(p);
            omp_set_schedule(omp_sched_static, 0);
            if (!images_close(reference, render_parallel(size, size, circles))) {
                std::printf("# %s baseline FAIL at %d threads, timings skipped\n",
                            scene.c_str(), p);
                valid = false;
                break;
            }
        }
        if (valid) {
            for (int p : threads) {
                omp_set_num_threads(p);
                if (!images_close(reference, render_parallel_optimized(
                                                  size, size, circles, tile, schedule, chunk,
                                                  &info))) {
                    std::printf("# %s optimized FAIL at %d threads, timings skipped\n",
                                scene.c_str(), p);
                    valid = false;
                    break;
                }
            }
        }
        if (!valid) {
            status = 1;
            continue;
        }

        const double c_avg = info.tile_count > 0
                                 ? static_cast<double>(info.candidate_entries) / info.tile_count
                                 : 0.0;
        const double t_avg = circles.empty()
                                 ? 0.0
                                 : static_cast<double>(info.candidate_entries) / circles.size();
        const double predicted = c_avg > 0.0 ? static_cast<double>(circles.size()) / c_avg : 0.0;

        std::printf("\n## scene %s: %zu circles, %dx%d, reference %.4f ms\n", scene.c_str(),
                    circles.size(), size, size, serial_ms);
        std::printf("# optimized lists: tile=%d tiles=%lld candidates=%lld c_avg=%.3f "
                    "t_avg=%.3f min_tile=%lld max_tile=%lld predicted_test_speedup=%.2fx\n",
                    info.tile_size, info.tile_count, info.candidate_entries, c_avg, t_avg,
                    info.min_candidates, info.max_candidates, predicted);
        std::printf("%-9s %3s %12s %12s %12s %9s %7s %8s %12s %12s\n", "variant", "p",
                    "median_ms", "min_ms", "max_ms", "speedup", "eff", "self", "pre_ms",
                    "render_ms");

        std::vector<Row> baseline_rows, optimized_rows;
        for (int p : threads) {
            Row row;
            row.scene = scene;
            row.variant = "baseline";
            row.schedule = "static";
            row.nc = static_cast<int>(circles.size());
            row.size = size;
            row.tile = 0;
            row.chunk = 0;
            row.threads = p;
            row.total = time_baseline(size, circles, p, runs);
            baseline_rows.push_back(row);
        }
        for (int p : threads) {
            Row row;
            row.scene = scene;
            row.variant = "optimized";
            row.schedule = schedule_name;
            row.nc = static_cast<int>(circles.size());
            row.size = size;
            row.tile = tile;
            row.chunk = chunk;
            row.threads = p;
            const OptimizedResult result = time_optimized(size, circles, p, runs, tile,
                                                          schedule, chunk);
            row.total = result.total;
            row.preprocess = result.preprocess.median;
            row.render = result.render.median;
            row.tile_count = result.info.tile_count;
            row.candidates = result.info.candidate_entries;
            row.c_avg = c_avg;
            row.t_avg = t_avg;
            row.predicted = predicted;
            optimized_rows.push_back(row);
        }

        const double t1_baseline = baseline_rows.front().total.median;
        const double t1_optimized = optimized_rows.front().total.median;
        for (Row& row : baseline_rows) {
            row.speedup = t1_baseline / row.total.median;
            row.self_speedup = row.speedup;
            row.efficiency = row.speedup / row.threads;
        }
        for (Row& row : optimized_rows) {
            row.speedup = t1_baseline / row.total.median;
            row.self_speedup = t1_optimized / row.total.median;
            row.efficiency = row.speedup / row.threads;
        }

        for (const Row& row : baseline_rows)
            std::printf("%-9s %3d %12.4f %12.4f %12.4f %9.3f %7.3f %8.3f %12s %12s\n",
                        row.variant.c_str(), row.threads, row.total.median,
                        row.total.minimum, row.total.maximum, row.speedup, row.efficiency,
                        row.self_speedup, "n/a", "n/a");
        for (const Row& row : optimized_rows)
            std::printf("%-9s %3d %12.4f %12.4f %12.4f %9.3f %7.3f %8.3f %12.4f %12.4f\n",
                        row.variant.c_str(), row.threads, row.total.median,
                        row.total.minimum, row.total.maximum, row.speedup, row.efficiency,
                        row.self_speedup, row.preprocess, row.render);

        const Row& last = optimized_rows.back();
        std::printf("# %s at p=%d: total %.4f = preprocess %.4f + render %.4f (+ %.4f "
                    "untimed in-call overhead)\n",
                    scene.c_str(), last.threads, last.total.median, last.preprocess,
                    last.render, last.total.median - last.preprocess - last.render);
        std::printf("# %s speedup vs baseline T1 = %.3fx; optimized self speedup = %.3fx\n",
                    scene.c_str(), last.speedup, last.self_speedup);

        // Per-thread busy-time diagnostic at the largest tested thread count.
        const int pmax = threads.back();
        omp_set_num_threads(pmax);
        RenderStats busy;
        busy.collect_thread_times = true;
        const Image busy_image =
            render_parallel_optimized(size, size, circles, tile, schedule, chunk, &busy);
        const bool busy_ok = images_close(reference, busy_image);
        if (!busy_ok) {
            std::printf("# %s busy diagnostic FAILED correctness, not reported\n",
                        scene.c_str());
            status = 1;
        } else {
            double sum_busy = 0.0, max_busy = 0.0, sum_span = 0.0, max_span = 0.0;
            for (int t = 0; t < pmax; ++t) {
                const double b = busy.thread_busy_ms[static_cast<std::size_t>(t)];
                const double s = busy.thread_span_ms[static_cast<std::size_t>(t)];
                sum_busy += b;
                sum_span += s;
                max_busy = std::max(max_busy, b);
                max_span = std::max(max_span, s);
            }
            const double mean_busy = sum_busy / pmax;
            const double mean_span = sum_span / pmax;
            std::printf("# busy p=%d: busy_ms mean=%.3f max=%.3f max/mean=%.2f | "
                        "span_ms mean=%.3f max=%.3f max/mean=%.2f | idle=%.1f%%\n",
                        pmax, mean_busy, max_busy, mean_busy > 0 ? max_busy / mean_busy : 0.0,
                        mean_span, max_span, mean_span > 0 ? max_span / mean_span : 0.0,
                        mean_span > 0 ? 100.0 * (1.0 - mean_busy / mean_span) : 0.0);
            for (int t = 0; t < pmax; ++t) {
                const double b = busy.thread_busy_ms[static_cast<std::size_t>(t)];
                const double s = busy.thread_span_ms[static_cast<std::size_t>(t)];
                std::printf("#   thread %d busy_ms=%.3f span_ms=%.3f\n", t, b, s);
                busy_rows.push_back(
                    BusyRow{scene, schedule_name, tile, chunk, pmax, t, b, s});
            }
        }

        rows.insert(rows.end(), baseline_rows.begin(), baseline_rows.end());
        rows.insert(rows.end(), optimized_rows.begin(), optimized_rows.end());
    }

    // CSV output, outside every timed region.
    std::ofstream csv("part4_results.csv");
    csv << "scene,nc,size,tile,schedule,chunk,variant,threads,median_ms,min_ms,max_ms,"
           "preprocess_ms,render_ms,speedup,self_speedup,efficiency,tile_count,"
           "candidate_entries,c_avg,t_avg,predicted_test_speedup\n";
    csv << std::fixed << std::setprecision(6);
    for (const Row& row : rows) {
        csv << row.scene << ',' << row.nc << ',' << row.size << ',' << row.tile << ','
            << row.schedule << ',' << row.chunk << ',' << row.variant << ',' << row.threads
            << ',' << row.total.median << ',' << row.total.minimum << ','
            << row.total.maximum << ',' << row.preprocess << ',' << row.render << ','
            << row.speedup << ',' << row.self_speedup << ',' << row.efficiency << ','
            << row.tile_count << ',' << row.candidates << ',' << row.c_avg << ','
            << row.t_avg << ',' << row.predicted << '\n';
    }

    std::ofstream busy_csv("part4_busy.csv");
    busy_csv << "scene,tile,schedule,chunk,threads,thread,busy_ms,span_ms\n";
    busy_csv << std::fixed << std::setprecision(6);
    for (const BusyRow& row : busy_rows)
        busy_csv << row.scene << ',' << row.tile << ',' << row.schedule << ',' << row.chunk
                 << ',' << row.threads << ',' << row.thread << ',' << row.busy << ','
                 << row.span << '\n';

    std::printf("\nwrote part4_results.csv (%zu rows) and part4_busy.csv (%zu rows)\n",
                rows.size(), busy_rows.size());
    return status;
}
