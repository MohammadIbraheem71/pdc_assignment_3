#!/usr/bin/env bash
# Builds and runs the Part 1 edge-case tests, saving output to saxpy_tests.log.
# Run from anywhere: ./saxpy/run_tests.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/saxpy/.part1_build"
mkdir -p "$build"

cd "$root"
g++ -O2 -std=c++17 -Wall -Wextra -fopenmp \
    saxpy/edge_tests.cpp saxpy/saxpy.cpp -o "$build/edge_tests"

"$build/edge_tests" | tee "$root/saxpy_tests.log"
