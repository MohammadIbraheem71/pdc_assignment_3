#!/usr/bin/env bash
# Builds and runs the Part 3 edge-case/determinism tests.
# Saves output to part3_tests.log.
# Run from anywhere: ./render/run_tests.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/render/.part3_build"
mkdir -p "$build"

cd "$root"
g++ -O2 -std=c++17 -Wall -Wextra -fopenmp \
    render/edge_tests.cpp render/renderer.cpp render/scenes.cpp -o "$build/edge_tests"

"$build/edge_tests" | tee "$root/part3_tests.log"
