#!/usr/bin/env bash
# Builds and runs the Part 2 benchmark.
# Saves the readable table to scan_bench.txt and the CSV to scan_results.csv.
#
# Usage: ./scan/run_bench.sh [runs] [threads] [sizes]
# Defaults: 9 runs, threads 1,2,4,8, sizes 100000,10000000.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/scan/.part2_build"
mkdir -p "$build"

cd "$root"
g++ -O2 -std=c++17 -Wall -Wextra -fopenmp \
    scan/bench_scan.cpp scan/scan.cpp -o "$build/bench_scan"

"$build/bench_scan" "$@" | tee "$root/scan_bench.txt"
