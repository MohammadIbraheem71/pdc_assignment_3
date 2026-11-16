// Part 1 benchmark for SAXPY. Separate from the supplied benchmark so that this
// version can emit CSV and use omp_get_wtime().
//
// Usage: ./bench_saxpy [runs] [threads] [sizes] [csv_path]
//   runs      timed runs per configuration, default 5
//   threads   comma separated thread counts, default 1,2,4,8
//   sizes     comma separated array lengths, default 10000,1000000,10000000
//   csv_path  output file, default saxpy_results.csv
//
// Every configuration gets one untimed warm-up call, then `runs` timed calls;
// the reported time is the median. T1 is saxpy_parallel() at one thread, so
// speedup Sp = T1/Tp and efficiency Ep = Sp/p. Output files are written only
// after all timing is finished.
#include "saxpy.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Row {
    unsigned long long n;
    int threads;
    double median_ms;
    double speedup;
    double efficiency;
    double serial_ms;
};

std::vector<unsigned long long> parse_list(const std::string& text) {
    std::vector<unsigned long long> values;
    std::stringstream in(text);
    std::string item;
    while (std::getline(in, item, ','))
        if (!item.empty()) values.push_back(std::strtoull(item.c_str(), nullptr, 10));
    return values;
}

// Always includes one thread first because it defines T1. Counts larger than
// the machine can run are skipped with a note, like the supplied benchmarks.
std::vector<int> usable_threads(const std::vector<unsigned long long>& requested) {
    const int procs = omp_get_num_procs();
    std::vector<int> result{1};
    for (unsigned long long p : requested) {
        if (p <= 1) continue;
        if (p > static_cast<unsigned long long>(procs)) {
            std::printf("# note: %llu threads skipped, only %d processors available\n", p, procs);
            continue;
        }
        if (std::find(result.begin(), result.end(), static_cast<int>(p)) == result.end())
            result.push_back(static_cast<int>(p));
    }
    std::sort(result.begin(), result.end());
    return result;
}

bool same_values(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::abs(a[i] - b[i]) > 0.0001f) return false;
    return true;
}

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

void print_table(const std::vector<Row>& rows) {
    std::printf("\n%-12s %7s %12s %9s %11s %12s %10s\n", "size", "threads", "median_ms",
                "speedup", "efficiency", "serial_ms", "vs_serial");
    for (const Row& row : rows) {
        std::printf("%-12llu %7d %12.6f %9.3f %11.3f %12.6f %10.3f\n", row.n, row.threads,
                    row.median_ms, row.speedup, row.efficiency, row.serial_ms,
                    row.serial_ms / row.median_ms);
    }
}

bool write_csv(const char* path, const std::vector<Row>& rows) {
    std::ofstream out(path);
    if (!out) return false;
    out << "n,threads,median_ms,speedup,efficiency,serial_ms,vs_serial\n";
    out << std::fixed << std::setprecision(6);
    for (const Row& row : rows) {
        out << row.n << ',' << row.threads << ',' << row.median_ms << ',' << row.speedup << ','
            << row.efficiency << ',' << row.serial_ms << ',' << (row.serial_ms / row.median_ms)
            << '\n';
    }
    return static_cast<bool>(out);
}

}  // namespace

int main(int argc, char** argv) {
    const int runs = argc > 1 ? std::atoi(argv[1]) : 5;
    const std::vector<int> threads =
        usable_threads(parse_list(argc > 2 ? argv[2] : "1,2,4,8"));
    const std::vector<unsigned long long> sizes =
        parse_list(argc > 3 ? argv[3] : "10000,1000000,10000000");
    const char* csv_path = argc > 4 ? argv[4] : "saxpy_results.csv";
    if (runs < 1 || sizes.empty()) {
        std::fprintf(stderr, "usage: ./bench_saxpy [runs] [threads] [sizes] [csv_path]\n");
        return 2;
    }

    omp_set_dynamic(0);  // keep the requested thread count for every run
    volatile float sink = 0.0f;  // prevents the timed calls from being discarded

    std::printf("# compiler %s\n", __VERSION__);
    std::printf("# _OPENMP %d\n", _OPENMP);
    std::printf("# processors available %d, default maximum threads %d\n", omp_get_num_procs(),
                omp_get_max_threads());
    std::printf("# runs %d timed + 1 warm-up, median reported, omp_get_wtime, I/O excluded\n", runs);
    std::printf("# T1 = saxpy_parallel on 1 thread; speedup = T1/Tp; efficiency = speedup/p\n");
    std::printf("# serial_ms = saxpy_serial median; vs_serial = serial_ms/median_ms\n");

    std::vector<Row> rows;

    for (unsigned long long n : sizes) {
        std::vector<float> x(static_cast<std::size_t>(n));
        std::vector<float> y(static_cast<std::size_t>(n));
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] = float(i % 97) / 10.0f;
            y[i] = float(i % 17);
        }
        const float a = 2.5f;
        const std::vector<float> expected = saxpy_serial(a, x, y);

        const double serial_ms = median_ms(
            [&] {
                const std::vector<float> r = saxpy_serial(a, x, y);
                if (!r.empty()) sink += r[0];
            },
            runs);

        double t1_ms = 0.0;
        for (int p : threads) {
            omp_set_num_threads(p);
            if (!same_values(expected, saxpy_parallel(a, x, y))) {
                std::printf("# n=%llu FAIL at %d threads, timings skipped\n", n, p);
                return 1;
            }
            const double ms = median_ms(
                [&] {
                    const std::vector<float> r = saxpy_parallel(a, x, y);
                    if (!r.empty()) sink += r[0];
                },
                runs);
            if (p == 1) t1_ms = ms;
            const double speedup = t1_ms / ms;
            rows.push_back(Row{n, p, ms, speedup, speedup / p, serial_ms});
        }
    }

    print_table(rows);
    if (!write_csv(csv_path, rows)) {
        std::fprintf(stderr, "error: cannot write CSV file '%s'\n", csv_path);
        return 2;
    }
    std::printf("\n# wrote %zu rows to %s\n", rows.size(), csv_path);
    (void)sink;
    return 0;
}
