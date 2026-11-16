#!/usr/bin/env bash
# Builds and runs the Part 2 edge-case tests, saving output to part2_tests.log.
# Run from anywhere: ./scan/run_tests.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/scan/.part2_build"
mkdir -p "$build"

cd "$root"
g++ -O2 -std=c++17 -Wall -Wextra -fopenmp \
    scan/edge_tests.cpp scan/scan.cpp -o "$build/edge_tests"

"$build/edge_tests" | tee "$root/part2_tests.log"
