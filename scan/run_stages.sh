#!/usr/bin/env bash
# Builds and runs the staged n=16 tree-scan demonstration.
# Saves output to part2_stages.txt.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/scan/.part2_build"
mkdir -p "$build"

cd "$root"
g++ -O2 -std=c++17 -Wall -Wextra scan/scan_stages.cpp -o "$build/scan_stages"

"$build/scan_stages" | tee "$root/part2_stages.txt"
