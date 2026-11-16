#!/usr/bin/env bash
# Builds and runs the Part 3 ordered-blending calculation check.
# Saves output to part3_blend.txt.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/render/.part3_build"
mkdir -p "$build"

cd "$root"
g++ -O2 -std=c++17 -Wall -Wextra -fopenmp \
    render/blend_check.cpp render/renderer.cpp -o "$build/blend_check"
OMP_NUM_THREADS=4 "$build/blend_check" | tee "$root/part3_blend.txt"
