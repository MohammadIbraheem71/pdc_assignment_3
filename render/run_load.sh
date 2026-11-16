#!/usr/bin/env bash
# Builds and runs the per-thread load probe on the clustered scene.
# Saves output to part3_load.txt.
# Usage: ./render/run_load.sh [size] [scene] [threads]
# With no arguments it runs 1024x1024 clustered at 8 and then 4 threads.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/render/.part3_build"
mkdir -p "$build"

cd "$root"
g++ -O2 -std=c++17 -Wall -Wextra -fopenmp \
    render/load_probe.cpp render/scenes.cpp -o "$build/load_probe"

if [ "$#" -eq 0 ]; then
    {
        "$build/load_probe" 1024 clustered 8
        echo
        "$build/load_probe" 1024 clustered 4
    } | tee "$root/part3_load.txt"
else
    "$build/load_probe" "$@" | tee "$root/part3_load.txt"
fi
