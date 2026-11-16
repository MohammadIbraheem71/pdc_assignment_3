# Part 1 Plan: SAXPY and Parallel Work-Sharing

## Goal

Implement `saxpy_parallel()` with an OpenMP work-sharing loop, verify it against the
serial reference on normal and edge-case inputs, measure speedup and efficiency at
1/2/4/8 threads for n = 1e4, 1e6, 1e7, and write up the measured results. Work is
limited to `saxpy/` plus new Part 1 files; the starter reference code, existing tests,
and signatures stay unchanged.

## Checklist

- [x] Read `README.md`, `Makefile`, `saxpy/saxpy.h`, `saxpy/saxpy.cpp`, `saxpy/main.cpp`, `saxpy/bench.cpp`, `common/timing.h`, `common/rng.h`
- [x] Implement `saxpy_parallel()` in `saxpy/saxpy.cpp`
- [x] Build with `make` and run `./saxpy_demo` (plus supplied benchmark correctness checks)
- [x] Add an edge-case test (separate file) and save its output to a text file
- [x] Write `saxpy/bench_saxpy.cpp` + run script (CSV + readable table, `omp_get_wtime`, warm-up + median of >=5)
- [x] Record CPU model, core count, OS, compiler version, OpenMP version / `_OPENMP`
- [x] Generate speedup-vs-threads PNG (matplotlib when available; Pillow fallback used here)
- [x] Write analysis notes (bandwidth limit, bytes/element, arithmetic intensity, measured efficiency drop)
- [x] Write `README_part1.md` with exact compile/test/benchmark commands
- [x] Run `make clean`, remove extra binaries, verify only report artifacts remain
- [x] Final summary (files, test status, timing table, manual follow-ups)

## Findings from reading the starter

- `Makefile`: `CXX ?= g++`, `CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -fopenmp`.
  `make` builds three demos and three benchmarks directly in the repo root (no objects).
  `make check` runs the demos and stops at the first failure. `make clean` removes the
  six binaries and `render_*.ppm`.
- `saxpy/saxpy.cpp`: `saxpy_serial()` is the untouched reference. `saxpy_parallel()`
  currently allocates a zero-filled output and returns it, so the demo reports FAIL.
  The comment notes the serial zero-fill of the output is inside the measured region.
- `saxpy/saxpy.h`: signatures are fixed; both functions return a new vector, throw
  `std::invalid_argument` on size mismatch, and accept empty inputs.
- `saxpy/main.cpp`: one deterministic input, compares with absolute tolerance `1e-4`,
  prints a single diagnostic timing, exit code 0/1. Not benchmark data.
- `saxpy/bench.cpp`: supplied benchmark; validates against serial before timing, uses
  `common/timing.h` (`std::chrono::steady_clock`), median after one warm-up, default
  5 runs. It prints a table but has no CSV and does not use `omp_get_wtime()`.
- `common/timing.h`: `usable_threads()` always puts 1 first and skips counts above
  `omp_get_num_procs()` with a `# note`; `median_ms()` warms up once then takes a median.
- `common/rng.h`: deterministic LCG, suitable for reproducible random test inputs.
- Thread counts must come from outside the implementation (`OMP_NUM_THREADS` for demos,
  `omp_set_num_threads()` in the bench).

## Decisions

- Implement plain `#pragma omp parallel for` over indices; each iteration writes its own
  `out[i]`, so no atomics or reductions are needed and element order is preserved.
- Keep the existing `std::vector<float> out(x.size())` allocation; constructor zero-fill
  is a measured cost shared with the reference, and avoiding it would add complexity.
- Do not modify the top-level `Makefile`: extra test/bench files are compiled directly
  with `g++` from small scripts under `saxpy/`, keeping `make` behavior untouched.
- New benchmark uses `omp_get_wtime()` for timing, 1 warm-up, 9 timed runs for the final
  measurement (>=5 required), median, `T1` = parallel code at 1 thread, `Sp = T1/Tp`,
  `Ep = Sp/p`. CSV and stdout table; all I/O happens after timing.
- Edge tests use `common/rng.h` with a fixed seed for a reproducible large array.
- `saxpy/probe_bandwidth.cpp` is an optional diagnostic (not part of the submitted
  implementation) that separates serial zero-fill cost from the parallel loop.
- Report artifacts live at the repo root: `saxpy_results.csv`, `saxpy_speedup.png`,
  `saxpy_tests.log`, `saxpy_bench.txt`; analysis goes in `part1_notes.md`.

## Findings / Results

- Implementation: `saxpy_parallel()` uses one `#pragma omp parallel for` over
  `std::size_t` indices; each iteration writes only `out[i]`. Output allocation and
  zero-fill are unchanged, so the benchmark measures the same footprint as the supplied
  implementation.
- Build: `make` succeeds with the default flags and no warnings from `saxpy/`.
- Demo: `SAXPY N=1000000 ... correctness=PASS`; `./saxpy_demo 0` also passes.
- Supplied benchmark spot check (`./saxpy_bench 1 1,2,4,8 10000`) passes correctness at
  every thread count. Its one-run timing is too noisy to interpret (2 threads showed
  speedup 1.135, 8 threads 1.017; efficiency 0.568 and 0.127).
- Machine (from `lscpu`, `uname`, `g++`, `OMP_DISPLAY_ENV`): Intel Xeon E5-2650 v3
  @ 2.30 GHz; 10 physical cores, 20 logical CPUs (2 threads/core, one socket);
  25 MiB L3; 6.9 GiB RAM; Ubuntu 26.04.1 LTS, kernel 7.0.0-43-generic x86_64;
  g++ 15.2.0 (Ubuntu 15.2.0-16ubuntu1), libgomp 16. Compile-time `_OPENMP = 201511`
  (OpenMP 4.5); the runtime reports `_OPENMP = 202111` (OpenMP 5.1) via
  `OMP_DISPLAY_ENV=VERBOSE`. 8 threads are supported. Load average was ~1.6 and the
  machine showed memory pressure (5.7/6.9 GiB used, swap in use), so absolute numbers
  are machine/load specific; the shape was stable over repeated runs.
- Tests: `saxpy/edge_tests.cpp` (run via `saxpy/run_tests.sh`) covers empty, single
  element, n=3 with up to 8 threads, n=7, n=1001, a 1e7 deterministic random array, and
  the size-mismatch exception. All 25 checks pass at 1/2/4/8 threads; output saved in
  `saxpy_tests.log`. Parallel results are within `1e-4` of the reference and inputs are
  unchanged.
- Benchmarks: final measurement is `./saxpy/run_bench.sh 9 1,2,4,8
  10000,1000000,10000000`; raw CSV in `saxpy_results.csv`, table in `saxpy_bench.txt`.

  | n | threads | median ms | speedup | efficiency | serial ms | vs serial |
  |---:|---:|---:|---:|---:|---:|---:|
  | 10000 | 1 | 0.009908 | 1.000 | 1.000 | 0.009174 | 0.926 |
  | 10000 | 2 | 0.009582 | 1.034 | 0.517 | 0.009174 | 0.957 |
  | 10000 | 4 | 0.011946 | 0.829 | 0.207 | 0.009174 | 0.768 |
  | 10000 | 8 | 0.012415 | 0.798 | 0.100 | 0.009174 | 0.739 |
  | 1000000 | 1 | 1.587167 | 1.000 | 1.000 | 1.552632 | 0.978 |
  | 1000000 | 2 | 1.223516 | 1.297 | 0.649 | 1.552632 | 1.269 |
  | 1000000 | 4 | 1.093735 | 1.451 | 0.363 | 1.552632 | 1.420 |
  | 1000000 | 8 | 1.079484 | 1.470 | 0.184 | 1.552632 | 1.438 |
  | 10000000 | 1 | 40.668898 | 1.000 | 1.000 | 42.224660 | 1.038 |
  | 10000000 | 2 | 40.678585 | 1.000 | 0.500 | 42.224660 | 1.038 |
  | 10000000 | 4 | 38.110118 | 1.067 | 0.267 | 42.224660 | 1.108 |
  | 10000000 | 8 | 38.373547 | 1.060 | 0.132 | 42.224660 | 1.100 |

- Diagnostic probe (`saxpy/probe_bandwidth.cpp`, 9-run medians, not part of the graded
  implementation) at n=1e7: allocate+zero-fill only = 34.67 ms; preallocated in-place
  loop = 13.66 / 12.06 / 11.79 / 11.75 ms at 1/2/4/8 threads; full implementation =
  42.15 / 41.70 / 40.08 / 39.91 ms. At n=1e6: zero-fill 0.488 ms, in-place 1.000 /
  0.607 / 0.341 / 0.163 ms, full 1.973 / 1.453 / 1.252 / 1.241 ms. At n=1e4: zero-fill
  0.0024 ms, in-place 0.0119 / 0.0077 / 0.0054 / 0.0050 ms, full 0.0134 / 0.0112 /
  0.0097 / 0.0128 ms. This isolates the serial allocate-and-zero-initialize step as the
  main limiter at 1e7, while the in-place loop itself scales only 1.16x (DRAM bandwidth).
- Analysis: `part1_notes.md` contains the bytes/element and arithmetic-intensity
  calculations, the measured bandwidth evidence, both efficiency-drop cases, and a
  separate assumptions/limitations section. `README_part1.md` lists exact commands.
- Final verification: `./saxpy_demo` and `./saxpy_demo 0` report `correctness=PASS`;
  the supplied benchmark validates at 1/2/4/8 threads; the edge suite reports
  `25 checks, 0 failed -> PASS`.
- Cleanup: `make clean` removed the six binaries and any `render_*.ppm`;
  `rm -rf saxpy/.part1_build` removed the extra test/benchmark binaries. No compiled
  executables or generated images remain apart from `saxpy_speedup.png`.
- Graph: matplotlib is not installed (`ModuleNotFoundError`), so
  `saxpy/plot_speedup.py` uses its Pillow fallback and wrote `saxpy_speedup.png`
  (1200x800) from `saxpy_results.csv`. The script uses matplotlib automatically on a
  machine that has it.

## Next steps

1. Re-run the final verification.
2. Run `make clean`, remove `saxpy/.part1_build`, verify no binaries/PPMs remain.
3. Report the final summary.

---

# Part 2: Exclusive Scan and Repeated-Index Compaction

## Goal

Implement `exclusive_scan_parallel()` and `find_repeats_parallel()` in `scan/scan.cpp`
without changing the serial references, signatures, or supplied tests. Verify against
the serial versions on the required edge lengths and random/patterned inputs, visualize
the tree scan stages for n=16, benchmark serial vs parallel scan (and repeats) at
n=1e5 and n=1e7 on 1/2/4/8 threads, and explain the measured behaviour using the
work/span model. Part 1 (`saxpy/`) is finished and must not be changed.

## Checklist

- [x] Read `README.md`, `Makefile`, `scan/scan.h`, `scan/scan.cpp`, `scan/main.cpp`, `scan/bench.cpp`, and the shared timing/RNG headers
- [x] Implement `exclusive_scan_parallel()` (tree-based up-sweep/down-sweep, padding to a power of two)
- [x] Implement `find_repeats_parallel()` using flags, scan offsets, and scatter
- [x] Build and run `./scan_demo` for the required lengths; record PASS/FAIL
- [x] Add a separate edge-case test file and save its output to a text file
- [x] Write the staged n=16 up/down-sweep program and save its output
- [x] Write the n=1025 padding calculation and strategy in the notes
- [x] Write `scan/bench_scan.cpp` + run script (CSV + readable table, `omp_get_wtime`, warm-up + median of >=5, both scan and repeats)
- [x] Record machine info (already in Part 1; refresh compiler/CPU/OpenMP facts for this part)
- [x] Generate a speedup-vs-threads PNG for both sizes
- [x] Write `part2_notes.md` (work/span, synchronization, padding, measured comparison, assumptions)
- [x] Write `README_part2.md` with exact compile/test/benchmark commands
- [x] Run `make clean`, remove extra binaries, verify only report artifacts remain
- [x] Final summary (files, test status, staged table, timing table, manual follow-ups)

## Findings from reading the starter

- `Makefile`: builds `scan_demo` from `scan/main.cpp` + `scan/scan.cpp`, and
  `scan_bench` from `scan/bench.cpp` + `scan/scan.cpp`, with
  `-O2 -std=c++17 -Wall -Wextra -fopenmp`. Extra Part 2 files will be compiled by
  scripts so the shared `Makefile` stays untouched.
- `scan/scan.cpp`: the serial scan and serial repeats functions are complete and must
  not be edited. Both parallel functions are stubs: `exclusive_scan_parallel()` returns
  all zeros and `find_repeats_parallel()` returns an empty vector.
- `scan/scan.h`: signatures are fixed. The scan result has the same length as the input
  and `output[0] = 0`; `< 2` elements produce no repeats. The comment explicitly says
  `find_repeats_parallel` should use `exclusive_scan_parallel` or the same scan
  technique on a 0/1 flag array.
- `scan/main.cpp`: `./scan_demo [n]` compares both functions on one deterministic input,
  exits 0 only if both match, and the README requires n = 0, 1, 7, 8, 9, 1024, 1025.
- `scan/bench.cpp`: supplied benchmark validates every parallel result against serial
  before timing, uses `common/timing.h` and the deterministic `Lcg` (range 0..3, seed
  12345), and reports medians. It has no CSV and does not use `omp_get_wtime()`.
- Machine (from Part 1): Intel Xeon E5-2650 v3, 10 physical cores / 20 logical CPUs,
  25 MiB L3, Ubuntu 26.04.1, g++ 15.2.0, libgomp 16, `_OPENMP = 201511`,
  runtime advertises OpenMP 5.1. 8 threads are supported.

## Decisions

- Use a Blelloch tree scan: copy the input into a power-of-two tree, up-sweep to
  inclusive subtree sums, set the root to 0, then down-sweep to exclusive prefixes and
  truncate to n. Reason: it makes each level's writes disjoint, directly matches the
  n=16 staged demonstration, and handles non-power-of-two lengths safely by padding
  with the identity element 0.
- Pad with zeros to `N = next_power_of_two(n)`. Padding adds zeros to every sum, so it
  cannot change any prefix value; the result is truncated back to n. This is simple and
  correct but costs memory and work at n=1025 (N=2048) and especially n=1e7
  (N=16777216). A blocked two-pass scan would avoid that padding; it is noted as the
  main alternative in the analysis.
- Use one `#pragma omp parallel` region with an `omp for` per level, relying on the
  implicit barrier at the end of each `omp for` and after a `single` that sets the root
  to zero. This avoids one fork/join per level while still synchronizing every stage.
- Allocate the padded tree with `std::unique_ptr<int[]>` (`new int[N]`) and initialize
  it in a parallel loop (`input[i]` for i<n, 0 otherwise), so there is no serial
  zero-fill of the whole tree. Build the returned `std::vector<int>` with the range
  constructor from the first n tree entries, avoiding another zero-initialized vector.
  Inputs are only read, never modified.
- `find_repeats_parallel`: build flags for i in [0, n-2] (so A[i+1] is never read out of
  bounds), scan the flags, compute `total = offsets.back() + flags.back()`, allocate the
  exact output size, and scatter `i` to `out[offsets[i]]`. Prefix offsets are strictly
  increasing at flagged positions, so the scatter has no write races and preserves
  ascending order.
- Extra test/benchmark/stage/plot programs live under `scan/` and are built by scripts;
  the top-level `Makefile` is unchanged.
- Benchmark uses 1 warm-up + 9 timed runs (>= 5 required), `omp_get_wtime()`, median,
  n = 1e5 and 1e7, 1/2/4/8 threads where supported. `T1` is the parallel function at 1
  thread, the serial reference is reported separately, `Sp = T1/Tp`, `Ep = Sp/p`.
  Both scan and repeats are measured.

## Findings / Results

- Implementation: `exclusive_scan_parallel()` pads to `N = next_power_of_two(n)` using a
  `std::unique_ptr<int[]>` tree initialized in parallel, runs the up-sweep and
  down-sweep inside one `omp parallel` region with one `omp for` per level and an
  implicit barrier between levels, sets the root to 0 under `omp single`, and returns
  `std::vector<int>(tree, tree+n)` (truncating the padding). n=0 returns `{}` and n=1
  returns `{0}` directly. `find_repeats_parallel()` builds the n-1 flags, calls the
  scan, computes the exact output length from the last offset plus last flag, and
  scatters the indices in parallel.
- Build: `make scan_demo` succeeds with no warnings.
- Demo: all seven required lengths (0, 1, 7, 8, 9, 1024, 1025) report
  `scan=PASS repeats=PASS` at OMP_NUM_THREADS = 1, 2, 4 and 8 (28/28 runs).
- Tests: `scan/edge_tests.cpp` (run via `scan/run_tests.sh`) covers the required lengths
  0/1/7/8/9/1024/1025, header examples, all-equal, all-distinct, alternating 0/1,
  alternating pairs, and fixed-seed random arrays of 2, 3, 17, 1025, 1e5, 1e6 and 1e7
  elements. All **84 checks pass** at 1/2/4/8 threads; inputs are unchanged; output is
  saved in `part2_tests.log`.
- Staged n=16 scan (`scan/scan_stages.cpp`, output in `part2_stages.txt`) matches the
  expected exclusive scan exactly. Levels:
  initial `[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]`;
  up d=1 `[1,3,3,7,1,3,3,7,1,3,3,7,1,3,3,7]`;
  up d=2 `[1,3,3,10,1,3,3,10,1,3,3,10,1,3,3,10]`;
  up d=4 `[1,3,3,10,1,3,3,20,1,3,3,10,1,3,3,20]`;
  up d=8 `[1,3,3,10,1,3,3,20,1,3,3,10,1,3,3,40]`;
  root set to 0 `[... , 0]`;
  down d=8 `[1,3,3,10,1,3,3,0,1,3,3,10,1,3,3,20]`;
  down d=4 `[1,3,3,0,1,3,3,10,1,3,3,20,1,3,3,30]`;
  down d=2 `[1,0,3,3,1,10,3,13,1,20,3,23,1,30,3,33]`;
  down d=1/final `[0,1,3,6,10,11,13,16,20,21,23,26,30,31,33,36]` (PASS).
- Benchmarks (`scan/bench_scan.cpp`, built and run by `scan/run_bench.sh`): 9 timed runs
  + 1 warm-up per point, median, `omp_get_wtime()`, n = 1e5 and 1e7, p = 1, 2, 4, 8,
  both scan and repeats. Raw data `scan_results.csv`, table `scan_bench.txt`. `T1` is
  the parallel function on 1 thread; the serial reference is a separate row.
  - n=1e5 scan: serial 0.094254 ms; parallel 0.702160 / 0.342428 / 0.254396 /
    0.158572 ms at p = 1/2/4/8; Sp = 1 / 2.051 / 2.760 / 4.428; vs_serial =
    0.134 / 0.275 / 0.371 / 0.594.
  - n=1e5 repeats: serial 0.410349 ms; parallel 1.841915 / 1.238569 / 0.957811 /
    0.830097 ms; Sp = 1 / 1.487 / 1.923 / 2.219; vs_serial =
    0.223 / 0.331 / 0.428 / 0.494.
  - n=1e7 scan: serial 39.062097 ms; parallel 238.780226 / 206.969825 / 189.146230 /
    194.364105 ms; Sp = 1 / 1.154 / 1.262 / 1.229; vs_serial =
    0.164 / 0.189 / 0.207 / 0.201.
  - n=1e7 repeats: serial 35.670948 ms; parallel 314.212601 / 269.393017 / 257.763239 /
    235.890687 ms; Sp = 1 / 1.166 / 1.219 / 1.332; vs_serial =
    0.114 / 0.132 / 0.138 / 0.151.
  - Result: the parallel tree scan is correct but **slower than serial at every measured
    point**. It scales up to 4.43x (scan, 1e5) from 1 to 8 threads but starts from 7.45x
    the serial time; at 1e7 the preallocated sweep scales only 1.11x (174.4 -> 156.8 ms).
- Diagnostic probe (`scan/probe_scan.cpp`) decomposes the n=1e7 8-thread time: full
  188.8 ms, preallocated sweep 156.8 ms, tree allocate+parallel-fill+result-copy
  53.2 ms (~28% of the total). The probe's `full` matches the CSV within a few percent.
- Derived cache-line analysis (no counters): the array-layout tree touches 6,291,454
  distinct 64-byte lines per sweep at N=2^24, 12,582,908 for both sweeps; at 64 B read +
  64 B write-back that is ~1.61 GB, i.e. ~10 GB/s over the measured 156.8 ms, versus
  ~80 MB for the serial scan. This memory traffic, not the add count, explains why the
  parallel version loses. Labelled as a derived estimate in the notes.
- Padding: n=1025 -> N=2048, 1023 extra elements = 99.805% (assignment text rounds to
  ~99.9%). n=1e5 -> 131072 (31.1% extra, 0.50 MiB); n=1e7 -> 16777216 (67.8% extra,
  64 MiB, 2.7x the 25 MiB L3).
- Graph: matplotlib is not installed (`ModuleNotFoundError`), so
  `scan/plot_scan_speedup.py` used its Pillow fallback and wrote `scan_speedup.png`
  (1200x800) from `scan_results.csv`; the script uses matplotlib automatically when
  available.
- Notes: `part2_notes.md` contains the staged table, work/span `2(N-1)` / `2 log2 N`,
  the barrier argument (48+2 barriers at n=1e7, 34+2 at n=1e5), the padding strategy,
  the measured tables, the probe decomposition, the cache-line model, and a separate
  "assumptions and limitations" section.
- Commands: `README_part2.md` lists the exact compile, test, staged-scan, benchmark,
  plot, and cleanup commands.
- Repeatability: one separate 1-run check of the supplied benchmark during a higher-load
  window measured the n=1e5 serial scan at 0.294 ms instead of 0.094 ms (an interference
  outlier on that core; the repeats serial row agreed). The 9-run medians in
  `scan_results.csv` are the recorded values; the n=1e7 gap (vs_serial ~0.20) is large and
  stable.
- Cleanup: `make clean` and `rm -rf scan/.part2_build` remove all executables; only the
  report artifacts (`scan_results.csv`, `scan_bench.txt`, `scan_speedup.png`,
  `part2_tests.log`, `part2_stages.txt`, `part2_notes.md`, `README_part2.md`) remain.

## Next steps

1. Keep the measured artifacts with the submission and put the staged table and timing
   table into the PDF report.
2. Write the explanations in own words (the notes are a reference, not a final report).
3. If a faster scan is wanted, replace the padded tree with a blocked two-pass scan
   (no padding, two linear passes) and re-measure; the tree version is the one required
   by the staged demonstration and is what was measured here.
4. Part 3 (renderer) is still untouched; `make check` will keep failing until it is done.

