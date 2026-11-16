# Part 1: SAXPY and Parallel Work-Sharing

Implementation, tests, benchmarks, and analysis for Part 1 only. The parallel function
is `saxpy_parallel()` in `saxpy/saxpy.cpp`; it uses one OpenMP `parallel for` loop.
Do not expect `make check` to pass yet: the supplied demo sequence also runs the
unfinished Part 2 and Part 3 stubs.

## Requirements

- Linux (or WSL) with `make` and a C++17 compiler that supports OpenMP (tested with
  g++ 15.2.0 / libgomp 16).
- Python 3 with Pillow for the graph here; if `matplotlib` is installed, the same
  script uses it automatically.

## Compile

From the repository root:

```sh
make                    # builds all demos and supplied benchmarks, including saxpy_demo
# or only the Part 1 supplied programs:
make saxpy_demo saxpy_bench
```

The Part 1 extra test and benchmark are compiled by their own scripts, so the starter
`Makefile` is left unchanged.

## Correctness tests

```sh
# Supplied demo: default 1e6 elements and an empty-input check
./saxpy_demo
./saxpy_demo 0

# Supplied benchmark correctness check (1 timed run is enough for validation)
./saxpy_bench 1 1,2,4,8 10000

# Extra edge-case tests: n=0, n=1, n<threads, non-divisible lengths, and 1e7 random.
# Writes the full output to saxpy_tests.log.
./saxpy/run_tests.sh
```

## Benchmark

```sh
# Final measurement: 9 timed runs + 1 warm-up, median, 1/2/4/8 threads,
# n = 1e4, 1e6, 1e7. Writes saxpy_results.csv and saxpy_bench.txt.
./saxpy/run_bench.sh 9 1,2,4,8 10000,1000000,10000000

# Argument order and defaults:
./saxpy/run_bench.sh [runs] [threads] [sizes]
```

The program prints and saves a table with `median_ms`, `speedup`, `efficiency`,
`serial_ms`, and `vs_serial`. `T1` is `saxpy_parallel` on one thread; `Sp = T1/Tp` and
`Ep = Sp/p`. Thread counts above the machine's processor count are skipped with a
`# note`, and every parallel result is checked against the serial reference before its
timing is recorded.

## Graph

```sh
python3 saxpy/plot_speedup.py saxpy_results.csv saxpy_speedup.png
```

Uses matplotlib when available; otherwise falls back to Pillow. The included PNG was
produced with the Pillow fallback because matplotlib was not installed on this machine.

## Optional diagnostic probe

Separates serial output zero-fill cost from the in-place parallel loop (not part of the
graded implementation):

```sh
g++ -O2 -std=c++17 -Wall -Wextra -fopenmp saxpy/probe_bandwidth.cpp -o /tmp/probe_bandwidth
/tmp/probe_bandwidth 10000000 9
/tmp/probe_bandwidth 1000000 9
/tmp/probe_bandwidth 10000 9
```

## Clean up

```sh
make clean                  # removes demos, supplied benchmarks, render_*.ppm
rm -rf saxpy/.part1_build   # removes the extra test/benchmark executables
```

Report artifacts to keep: `saxpy_results.csv`, `saxpy_speedup.png`,
`saxpy_tests.log`, `saxpy_bench.txt`, `plan.md`, and `part1_notes.md`.
