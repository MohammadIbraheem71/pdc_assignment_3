#!/usr/bin/env bash
# Builds and runs the Part 4 schedule/tile sweep and per-thread busy diagnostics.
# Saves the readable table to part4_sweep.txt. The program writes
# part4_sweep.csv and part4_sweep_busy.csv.
#
# Usage:
#   ./render/run_part4_sweep.sh [size] [runs] [threads] [scene[,scene...]]
# Defaults: 1024 5 8 p4_corner,p4_heavy
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/render/.part4_build"
mkdir -p "$build"

cd "$root"
g++ -O2 -std=c++17 -Wall -Wextra -fopenmp \
    render/sweep_part4.cpp render/part4_scenes.cpp render/scenes.cpp render/renderer.cpp \
    -o "$build/sweep_part4"

if [ "$#" -eq 0 ]; then
    set -- 1024 5 8 p4_corner,p4_heavy
fi

"$build/sweep_part4" "$@" | tee "$root/part4_sweep.txt"
