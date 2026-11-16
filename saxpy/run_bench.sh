#!/usr/bin/env bash
# Builds and runs the Part 1 SAXPY benchmark, saving the readable table to
# saxpy_bench.txt and the CSV to saxpy_results.csv (both in the repo root).
#
# Usage: ./saxpy/run_bench.sh [runs] [threads] [sizes]
# Defaults: 5 runs, threads 1,2,4,8, sizes 10000,1000000,10000000.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/saxpy/.part1_build"
mkdir -p "$build"

cd "$root"
g++ -O2 -std=c++17 -Wall -Wextra -fopenmp \
    saxpy/bench_saxpy.cpp saxpy/saxpy.cpp -o "$build/bench_saxpy"

"$build/bench_saxpy" "$@" | tee "$root/saxpy_bench.txt"
