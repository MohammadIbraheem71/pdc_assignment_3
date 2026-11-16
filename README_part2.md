# Part 2: Exclusive Scan and Repeated-Index Compaction

Implementation, tests, benchmarks, and analysis for Part 2 only. The parallel functions
are `exclusive_scan_parallel()` and `find_repeats_parallel()` in `scan/scan.cpp`. The
scan is a Blelloch tree scan padded to a power of two; repeats are built as
flags -> scan -> scatter. Part 1 (`saxpy/`) is unchanged. `make check` still stops at the
unfinished Part 3 renderer stub, which is expected at this stage.

## Requirements

- Linux (or WSL) with `make` and a C++17 compiler that supports OpenMP (tested with
  g++ 15.2.0 / libgomp 16).
- Python 3 with Pillow for the graph; if `matplotlib` is installed, the same script uses
  it automatically.

## Compile

From the repository root:

```sh
make scan_demo scan_bench     # the two supplied Part 2 programs
# or everything (still stops at the Part 3 demo when run):
make
```

The extra test, staged-scan, benchmark, and plot programs live in `scan/` and are built
by their own scripts, so the starter `Makefile` is left unchanged.

## Correctness tests

```sh
# Supplied demo at every required length and thread count.
# Each line must print scan=PASS repeats=PASS.
for t in 1 2 4 8; do
  for n in 0 1 7 8 9 1024 1025; do
    OMP_NUM_THREADS=$t ./scan_demo "$n"
  done
done

# Supplied benchmark's correctness check (validates before timing).
./scan_bench 1 1,2,4,8 100000

# Extra edge tests: required lengths, header examples, all-equal, all-distinct,
# alternating patterns, and fixed-seed random arrays up to 1e7.
# Writes the full output to part2_tests.log and prints the PASS/FAIL summary.
./scan/run_tests.sh
```

## Staged n = 16 tree-scan walkthrough

```sh
./scan/run_stages.sh          # writes part2_stages.txt, ends with STAGED SCAN: PASS
```

It prints the array after every up-sweep level, after setting the root to 0, after every
down-sweep level, and the final array. The final array is
`[0,1,3,6,10,11,13,16,20,21,23,26,30,31,33,36]`. The same table is in
`part2_notes.md`.

## Benchmark

```sh
# Final measurement: 9 timed runs + 1 warm-up, median, omp_get_wtime,
# threads 1,2,4,8, sizes 1e5 and 1e7, both scan and repeats.
# Writes scan_results.csv and the readable table scan_bench.txt.
./scan/run_bench.sh

# Same with explicit arguments:
./scan/run_bench.sh 9 1,2,4,8 100000,10000000
# Argument order and defaults:
./scan/run_bench.sh [runs] [threads] [sizes] [csv_path]
```

The table has `median_ms`, `speedup`, `efficiency`, `serial_ms`, and `vs_serial`. **T1 is
the parallel function on one thread**, not the serial reference; the serial reference is
its own row. `Sp = T1/Tp`, `Ep = Sp/p`, `vs_serial = serial_ms/Tp`. Each parallel result
is checked against the serial result before its timing is recorded. Thread counts above
the machine's processor count are skipped with a `# note`. No I/O is inside a timed
region.

## Graph

```sh
python3 scan/plot_scan_speedup.py scan_results.csv scan_speedup.png
```

Plots `Sp` vs threads for scan and repeats at both sizes, with the `S = 1` line. Uses
matplotlib when available; otherwise falls back to Pillow. The included PNG was produced
with the Pillow fallback because matplotlib was not installed on this machine.

## Optional diagnostic probe

Separates the per-call tree allocation/fill/result copy from the up/down sweep (not part
of the graded implementation):

```sh
g++ -O2 -std=c++17 -Wall -Wextra -fopenmp -I scan \
    scan/probe_scan.cpp scan/scan.cpp -o /tmp/probe_scan
/tmp/probe_scan 100000 9
/tmp/probe_scan 10000000 9
```

It prints `full_ms`, `sweep_ms` (preallocated tree), and `alloc_init_copy_ms` for
p = 1, 2, 4, 8.

## Clean up

```sh
make clean                  # removes demos, supplied benchmarks, render_*.ppm
rm -rf scan/.part2_build    # removes the extra test/stage/benchmark executables
```

Report artifacts to keep: `scan_results.csv`, `scan_speedup.png`, `scan_bench.txt`,
`part2_tests.log`, `part2_stages.txt`, `plan.md`, `part2_notes.md`.
