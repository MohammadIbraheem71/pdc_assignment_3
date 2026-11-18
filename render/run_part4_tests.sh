#!/usr/bin/env bash
# Builds and runs the Part 4 correctness/determinism tests.
# Saves output to part4_tests.log.
# Run from anywhere: ./render/run_part4_tests.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/render/.part4_build"
mkdir -p "$build"

cd "$root"
g++ -O2 -std=c++17 -Wall -Wextra -fopenmp \
    render/part4_tests.cpp render/part4_scenes.cpp render/scenes.cpp render/renderer.cpp \
    -o "$build/part4_tests"

"$build/part4_tests" | tee "$root/part4_tests.log"
