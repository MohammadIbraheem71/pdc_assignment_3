#!/usr/bin/env bash
# Builds and runs the Part 4 main benchmark, saving the readable table to
# part4_bench.txt. The program itself writes part4_results.csv and part4_busy.csv.
#
# Usage:
#   ./render/run_part4_bench.sh [size] [runs] [threads] [scenes] [tile] [schedule] [chunk]
# Defaults:
#   1024 5 1,2,4,8 p4_sparse,p4_heavy,p4_corner,p4_few,p4_one,p4_huge 16 static 0
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/render/.part4_build"
mkdir -p "$build"

cd "$root"
g++ -O2 -std=c++17 -Wall -Wextra -fopenmp \
    render/bench_part4.cpp render/part4_scenes.cpp render/scenes.cpp render/renderer.cpp \
    -o "$build/bench_part4"

if [ "$#" -eq 0 ]; then
    set -- 1024 5 1,2,4,8 \
        p4_sparse,p4_heavy,p4_corner,p4_few,p4_one,p4_huge 16 static 0
fi

"$build/bench_part4" "$@" | tee "$root/part4_bench.txt"
