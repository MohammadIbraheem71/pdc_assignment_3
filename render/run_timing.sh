#!/usr/bin/env bash
# Builds and runs the Part 3 light timing sanity check.
# Saves output to part3_timing.txt.
# Usage: ./render/run_timing.sh [size] [runs] [scene]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/render/.part3_build"
mkdir -p "$build"

cd "$root"
g++ -O2 -std=c++17 -Wall -Wextra -fopenmp \
    render/timing_sanity.cpp render/renderer.cpp render/scenes.cpp -o "$build/timing_sanity"

"$build/timing_sanity" "$@" | tee "$root/part3_timing.txt"
