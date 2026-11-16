// Part 2 benchmark. Separate from the supplied benchmark so it can use
// omp_get_wtime() and emit CSV.
//
// Usage: ./bench_scan [runs] [threads] [sizes] [csv_path]
//   runs      timed runs per configuration, default 9
//   threads   comma separated thread counts, default 1,2,4,8
//   sizes     comma separated array lengths, default 100000,10000000
//   csv_path  output file, default scan_results.csv
//
// T1 is the parallel function at one thread. The serial reference is measured
// separately. speedup = T1/Tp and efficiency = speedup/p. Every parallel result
// is checked against the serial result before its timing is recorded. All I/O
// happens after the timed measurements.
#include "scan.h"
#include "../common/rng.h"

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
    std::string op;
    std::string variant;
    int threads;
    double median_ms;
    double speedup;
    double efficiency;
    double serial_ms;
};

volatile long long sink = 0;  // keeps the timed calls from being discarded

std::vector<unsigned long long> parse_list(const std::string& text) {
    std::vector<unsigned long long> values;
    std::stringstream in(text);
    std::string item;
    while (std::getline(in, item, ','))
        if (!item.empty()) values.push_back(std::strtoull(item.c_str(), nullptr, 10));
    return values;
}

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

bool same_values(const std::vector<int>& a, const std::vector<int>& b) {
    return a == b;
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

template <class Serial, class Parallel>
int bench_one(const char* op, unsigned long long n, const std::vector<int>& input,
              Serial serial, Parallel parallel, const std::vector<int>& threads, int runs,
              std::vector<Row>& rows) {
    const std::vector<int> expected = serial(input);

    auto time_call = [&](auto&& fn) {
        return median_ms(
            [&] {
                const std::vector<int> r = fn(input);
                if (!r.empty()) sink += r[0];
            },
            runs);
    };

    const double serial_ms = time_call(serial);
    rows.push_back(Row{n, op, "serial", 1, serial_ms, 0.0, 0.0, serial_ms});

    double t1_ms = 0.0;
    for (int p : threads) {
        omp_set_num_threads(p);
        if (!same_values(expected, parallel(input))) {
            std::printf("# n=%llu %s FAIL at %d threads, timings skipped\n", n, op, p);
            return 1;
        }
        const double ms = time_call(parallel);
        if (p == 1) t1_ms = ms;
        const double speedup = t1_ms / ms;
        rows.push_back(Row{n, op, "parallel", p, ms, speedup, speedup / p, serial_ms});
    }
    return 0;
}

void print_table(const std::vector<Row>& rows) {
    std::printf("\n%-12s %-10s %-9s %7s %12s %9s %11s %12s %10s\n", "size", "op", "variant",
                "threads", "median_ms", "speedup", "efficiency", "serial_ms", "vs_serial");
    for (const Row& row : rows) {
        if (row.variant == "serial") {
            std::printf("%-12llu %-10s %-9s %7d %12.6f %9s %11s %12.6f %10.3f\n", row.n,
                        row.op.c_str(), row.variant.c_str(), row.threads, row.median_ms, "n/a",
                        "n/a", row.serial_ms, 1.0);
        } else {
            std::printf("%-12llu %-10s %-9s %7d %12.6f %9.3f %11.3f %12.6f %10.3f\n", row.n,
                        row.op.c_str(), row.variant.c_str(), row.threads, row.median_ms,
                        row.speedup, row.efficiency, row.serial_ms, row.serial_ms / row.median_ms);
        }
    }
}

bool write_csv(const char* path, const std::vector<Row>& rows) {
    std::ofstream out(path);
    if (!out) return false;
    out << "n,op,variant,threads,median_ms,speedup,efficiency,serial_ms,vs_serial\n";
    out << std::fixed << std::setprecision(6);
    for (const Row& row : rows) {
        out << row.n << ',' << row.op << ',' << row.variant << ',' << row.threads << ','
            << row.median_ms;
        if (row.variant == "serial")
            out << ",,,";
        else
            out << ',' << row.speedup << ',' << row.efficiency << ',';
        out << row.serial_ms << ',' << (row.serial_ms / row.median_ms) << '\n';
    }
    return static_cast<bool>(out);
}

}  // namespace

int main(int argc, char** argv) {
    const int runs = argc > 1 ? std::atoi(argv[1]) : 9;
    const std::vector<int> threads = usable_threads(parse_list(argc > 2 ? argv[2] : "1,2,4,8"));
    const std::vector<unsigned long long> sizes =
        parse_list(argc > 3 ? argv[3] : "100000,10000000");
    const char* csv_path = argc > 4 ? argv[4] : "scan_results.csv";
    if (runs < 1 || sizes.empty()) {
        std::fprintf(stderr, "usage: ./bench_scan [runs] [threads] [sizes] [csv_path]\n");
        return 2;
    }

    omp_set_dynamic(0);

    std::printf("# compiler %s\n", __VERSION__);
    std::printf("# _OPENMP %d\n", _OPENMP);
    std::printf("# processors available %d, default maximum threads %d\n", omp_get_num_procs(),
                omp_get_max_threads());
    std::printf("# runs %d timed + 1 warm-up, median reported, omp_get_wtime, I/O excluded\n", runs);
    std::printf("# T1 = parallel function on 1 thread; speedup = T1/Tp; efficiency = speedup/p\n");
    std::printf("# serial_ms = serial reference median; vs_serial = serial_ms/median_ms\n");

    std::vector<Row> rows;
    int status = 0;

    for (unsigned long long n : sizes) {
        Lcg rng(12345);
        std::vector<int> input(n);
        for (std::size_t i = 0; i < input.size(); ++i) input[i] = rng.next_int(4);

        status |= bench_one("scan", n, input,
                            [](const std::vector<int>& v) { return exclusive_scan_serial(v); },
                            [](const std::vector<int>& v) { return exclusive_scan_parallel(v); },
                            threads, runs, rows);
        status |= bench_one("repeats", n, input,
                            [](const std::vector<int>& v) { return find_repeats_serial(v); },
                            [](const std::vector<int>& v) { return find_repeats_parallel(v); },
                            threads, runs, rows);
    }

    print_table(rows);
    if (!write_csv(csv_path, rows)) {
        std::fprintf(stderr, "error: cannot write CSV file '%s'\n", csv_path);
        return 2;
    }
    std::printf("\n# wrote %zu rows to %s\n", rows.size(), csv_path);
    (void)sink;
    return status;
}
