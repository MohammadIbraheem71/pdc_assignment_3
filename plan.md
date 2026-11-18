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

---

# Part 3: Correct Parallel Circle Rendering

## Goal

Implement `render_parallel()` in `render/renderer.cpp` so that it matches
`render_reference()` within the supplied tolerance on every scene, while assigning
disjoint pixels/rows to OpenMP workers and applying the circles to each pixel in their
original input order. Keep the implementation simple and correct: this is the Part 4
baseline, so no culling or bounding boxes here. Add edge-case and determinism tests,
produce example images, do a light timing sanity check, and write `part3_notes.md`
with the ordered-blending calculation, the lost-update race interleaving, the
decomposition discussion, and the 1024x1024 load calculation. Do not modify `saxpy/`,
`scan/`, the serial reference, signatures, or the supplied comparison code.

## Checklist

- [x] Read `README.md`, `Makefile`, `render/renderer.h`, `render/renderer.cpp`, `render/scenes.cpp`, `render/scenes.h`, `render/main.cpp`, `render/bench.cpp`, and the image comparison code
- [x] Implement `render_parallel()` (row-parallel, original circle order per pixel, one write per pixel, `schedule(runtime)`)
- [x] Build and run `./render_demo` for the supplied scenes; record PASS/FAIL
- [x] Add a separate edge-case + determinism test file and save its output to a text file
- [x] Write the exact scene-generation/example-image commands and save example images in a clearly named folder
- [x] Light timing sanity check: serial vs parallel at 1/2/4/8, warm-up + >=5 runs, median, `omp_get_wtime`, save to a text file
- [x] Record machine info (already in Parts 1 and 2; confirm it still applies)
- [x] Verify ordered-blending calculations with a small program and save its output
- [x] Measure per-thread work on a clustered 1024x1024 scene (probe) to support the load-imbalance discussion
- [x] Write `part3_notes.md` (blending math, race table, decomposition, load calculation, measured vs assumed)
- [x] Write `README_part3.md` with exact compile/test/run commands
- [x] Preserve the Part 3 baseline for Part 4 (git tag/commit + note in code/plan)
- [x] Run `make clean`, remove extra binaries, keep only the report images in their folder
- [x] Final summary (files, tests, blending results, allocation numbers, timing table, manual follow-ups)

## Findings from reading the starter

- `render/renderer.h`: `Circle` is seven floats (`x, y, r, red, green, blue, alpha`) in
  normalized [0,1] coordinates; `Pixel` is `float r,g,b`; `Image` is
  `{int width,height; std::vector<Pixel> pixels}` stored row by row, so pixel (x,y) is
  `pixels[y*width + x]`. `images_close(a,b,tolerance=0.0001f)` requires equal dimensions
  and every component difference `<= tolerance` on all channels.
- `render/renderer.cpp`: the reference starts from an all-white image
  (`Pixel{1,1,1}`), then for each circle in input order visits every pixel and calls
  `blend` when `covers` is true. `blend` computes
  `p.r = alpha*red + (1-alpha)*p.r` (same for g and b) with `float` arithmetic.
  `covers` computes the pixel center as `px=(x+0.5f)/w, py=(y+0.5f)/h`, then
  `dx=px-c.x, dy=py-c.y` and returns `dx*dx + dy*dy <= c.r*c.r`. The boundary is
  included (`<=`), and all types are `float`; the parallel version must reuse these
  exact helpers/expressions.
- `render/main.cpp`: `./render_demo [size] [scene] [variant]` renders the reference and
  the chosen variant, writes `render_<scene>_reference.ppm` and
  `render_<scene>_<variant>.ppm` into the current directory (even when the comparison
  fails), prints `serial_ms`, `parallel_ms`, and `correctness=PASS|FAIL`, and exits 1 on
  failure. Size is limited to 1..4096.
- `render/bench.cpp`: measures the serial reference, `render_parallel` ("baseline") and
  `render_optimized` at 1/2/4/8 threads, validating every parallel image before timing.
  It uses the shared `common/timing.h` (`median_ms`, `omp_get_wtime` is not used there but
  `steady_clock` is). It does not write images, so nothing is timed with I/O. While
  `render_optimized` is a stub, the benchmark reports a FAIL and exits 1, so Part 3 needs
  its own timing program.
- `render/scenes.cpp`: deterministic fixed-seed scenes: `dense` (40 circles on a grid,
  alpha 0.35), `sparse` (300 tiny circles, seed 1001), `overlap` (150 large low-opacity
  circles near the middle, seed 2002), `clustered` (500 small circles in two hot spots,
  seed 3003). The scene names/order are fixed by `scenes.h`.
- `Makefile`: builds `render_demo` from `render/main.cpp render/renderer.cpp
  render/scenes.cpp` and `render_bench` from `render/bench.cpp` plus the same files, with
  `-O2 -std=c++17 -Wall -Wextra -fopenmp`. `make clean` removes all six binaries and
  `render_*.ppm` in the repository root. The Makefile has no wildcard rules, so extra
  Part 3 programs will be compiled by scripts and the Makefile will not be changed.
- Machine (from Parts 1 and 2): Intel Xeon E5-2650 v3 @ 2.30 GHz, 10 physical cores /
  20 logical CPUs, 25 MiB L3, 6.9 GiB RAM, Ubuntu 26.04.1 LTS (kernel
  7.0.0-43-generic), g++ 15.2.0, libgomp 16, `_OPENMP = 201511` (OpenMP 4.5), runtime
  advertises OpenMP 5.1. 8 threads are supported; the machine is shared/loaded.

## Decisions

- Decompose by **rows**: `#pragma omp parallel for schedule(runtime)` over `y`, with the
  inner `x` loop and the circle loop inside. Every pixel is written by exactly one
  iteration, rows are independent, and `schedule(runtime)` lets Part 4 use
  `OMP_SCHEDULE` experiments. Static assignment of rows is the default schedule.
- Keep the pixel color in a local `Pixel` during the circle loop (`p = {1,1,1}`), apply
  the circles in input order 0..N-1 with the existing `covers()` and `blend()` helpers,
  then write `image.pixels[y*width + x] = p` once. No circle is reordered, no atomics, no
  locks. Reusing the reference helpers guarantees identical pixel-center, float, and
  `<=` boundary semantics.
- Do not call `render_reference` and do not set the thread count inside the function. The
  initial image is constructed all-white exactly like the reference; no I/O happens in
  the renderer.
- The simple row loop is the baseline for Part 4. `render_parallel` will not be modified
  in Part 4; a git tag (`part3-baseline`) plus a comment above the function will mark it
  so the optimized version can be compared against it.
- Extra files live under `render/` and are built by scripts: edge/determinism tests,
  a timing-sanity program, a per-thread work probe, and an example-image script. The
  supplied tests, scenes, reference, and comparison remain untouched.
- Example images go into `part3_images/` (a clearly named folder), with a deterministic
  script that renders the same fixed-seed scenes every time. `make clean` only removes
  `render_*.ppm` in the repository root, so the report folder is unaffected.
- Timing sanity check: one fixed scene, serial reference vs `render_parallel` at 1/2/4/8
  threads, one warm-up + at least 5 timed runs, median, `omp_get_wtime()`, no I/O in the
  timed region. Full benchmarking and schedule experiments belong to Part 4.
- `part3_notes.md` will keep measured numbers and derived arithmetic clearly separate
  from reasoning/assumptions, and the per-thread imbalance claim will be supported by a
  measured probe if it is cheap enough to run.

## Findings / Results

- Implementation: `render_parallel()` decomposes by rows
  (`#pragma omp parallel for schedule(runtime)`, y outer, x inner, circles inner-most in
  original order). Each pixel keeps a local `Pixel p{1,1,1}`, calls the existing
  `covers()`/`blend()` in order, and writes `image.pixels[y*width+x]` once. No atomics,
  no locks, no circle reordering, no thread-count changes, no I/O.
- Build: `make render_demo` succeeds with no warnings.
- Demo: all four supplied scenes (`dense`, `sparse`, `overlap`, `clustered`) at 256x256
  and `OMP_NUM_THREADS` = 1, 2, 4, 8 report `correctness=PASS` (16/16 runs). Single-run
  diagnostics were 9–105 ms serial and 2.5–60 ms parallel, so the timing check below
  uses medians.
- Edge/determinism tests: `render/edge_tests.cpp` (run by `render/run_tests.sh`) covers
  zero circles, 0x0 and 0x7 images, non-square images, single circles (centre/corner/
  edge), mostly-empty scenes, overlapping A-then-B and B-then-A, partly and fully
  outside circles, 64 concentric circles with different colors/opacities, and a mixed
  120-circle random scene. All **56 checks pass** at 1/2/4/8 threads; every result is
  both within `images_close` (1e-4) and **bit-exact** against the reference, and four
  re-renders per case are identical. Output: `part3_tests.log`.
- Timing sanity (`render/timing_sanity.cpp`, `render/run_timing.sh`): 512x512 clustered
  scene, 500 circles, 7 timed runs + 1 warm-up, median with `omp_get_wtime()`, no I/O.
  Serial reference 333.5046 ms; parallel 189.2990 / 100.9235 / 54.8234 / 28.2888 ms at
  p = 1/2/4/8. `Sp = T1/Tp` = 1 / 1.876 / 3.453 / 6.692, efficiency
  1 / 0.938 / 0.863 / 0.837, `vs_serial` = 1.762 / 3.305 / 6.083 / 11.789. Output:
  `part3_timing.txt`. A repeat run in the same session gave 323.7280 ms serial and
  187.1182 / 96.5131 / 52.3922 / 28.3514 ms parallel (`Sp` = 1 / 1.939 / 3.571 / 6.600),
  so the shape is stable.
  - The parallel function at **1 thread is 1.76x faster than the serial reference**:
    both do the same per-pixel coverage tests and blends in the same order, but the
    reference does a load+store of the image pixel for every covered circle, while the
    baseline keeps the pixel in a local variable and writes once. The scaling numbers are
    therefore reported both against T1 (parallel at 1 thread) and against the serial
    reference.
- Ordered blending check (`render/blend_check.cpp`, `render/run_blend.sh`): A then B gives
  (0.500000, 0.250000, 0.750000), B then A gives (0.750000, 0.250000, 0.500000); the same
  values come out of both `render_reference` and `render_parallel` at the centre pixel
  of a 100x100 image. Blending is not commutative. Output: `part3_blend.txt`.
- Load probe (`render/load_probe.cpp`, `render/run_load.sh`): 1024x1024 clustered, 500
  circles, 21.4% pixels covered, 1,077,826 blends. At 8 threads each thread gets exactly
  128 rows / 131,072 pixels, but blend counts range from 17,955 to 363,211 (**20.2x**);
  measured per-thread loop times range from 146.150 to 164.587 ms (**1.13x**). At 4
  threads (256 rows each) blend counts range from 62,447 to 383,214 (6.1x) and loop times
  from 285.523 to 306.701 ms (1.07x). So blend work is very uneven, but the baseline's
  time is dominated by the uniform 500-circle test per pixel (524,288,000 tests), of
  which only 0.21% lead to a blend. Output: `part3_load.txt`.
- Example images: `render/run_examples.sh` renders the fixed-seed `dense` and `clustered`
  scenes at 512x512 and writes reference/parallel PNGs (converted losslessly from the
  starter's text PPM) into `part3_images/`:
  `dense_512_{reference,parallel}.png` and `clustered_512_{reference,parallel}.png`.
  Each reference/parallel pair has identical bytes, as expected from the bit-exact tests.

- `part3_notes.md` written: implementation, verification summary, measured timing and
  load tables, ordered-blending working, lost-update race interleaving (including the
  finer mixed-channel case), circle- vs pixel-parallel discussion with lock/false-sharing
  reasoning, 1024x1024 load calculation, and a separate measured-vs-assumed section.
- `README_part3.md` written with exact compile, demo, test, blend, timing, load, example,
  and cleanup commands, plus the Part 4 baseline note.
- Final verification: after the Part 3 commit, `make render_demo` and all four scenes at
  384x384 reported `correctness=PASS` at 1 and 8 threads (8/8 runs), and
  `./render/run_tests.sh` again reported `56 checks, 0 failed -> PASS`.
- Baseline preserved: Part 3 is committed and tagged `part3-baseline`; `render_parallel()`
  carries a comment that it is the frozen baseline, and Part 4 should only implement
  `render_optimized()`.
- Cleanup: `make clean` removed the six binaries and all `render_*.ppm` files;
  `rm -rf render/.part3_build` removed the extra test/benchmark binaries. Root contains no
  executables or PPMs. Kept for the report: `part3_images/` (4 PNGs), `part3_tests.log`,
  `part3_timing.txt`, `part3_load.txt`, `part3_blend.txt`, `part3_notes.md`,
  `README_part3.md`. Delete `part3_images/` before zipping the submission.

## Next steps

1. Re-run the final demo and edge tests from the committed tree.
2. `make clean`, remove `render/.part3_build` and the leftover root PPMs, verify that only
   report artifacts (`part3_*` files, `part3_images/`, `part3_notes.md`,
   `README_part3.md`) remain.
3. Produce the final summary for the user.

---

# Part 4: Renderer Optimization and Performance Investigation

## Goal

Implement an optimized parallel renderer that produces exactly the same image as the
Part 3 baseline/reference while skipping work that cannot change the result, then measure
where the win comes from and where it disappears. The chosen primary technique is
**Option B: ordered per-tile candidate lists** (CSR-style), with a conservative padded
circle bounding box used only as a filter and the exact reference `covers()`/`blend()`
used per pixel. `render_parallel()` stays untouched as the baseline (tag
`part3-baseline`); the optimized code is a new function selected by the benchmark driver.

The investigation must cover deterministic scenes (sparse, heavy overlap, unevenly
clustered, and pathological few/huge circles), correctness at several thread counts and
tile sizes, median benchmark tables, a schedule/tile-size sweep, per-thread busy-time
data, graphs, and a work-model comparison.

## Checklist

- [x] Re-read the Part 3 code, plan, and notes; write down the baseline cost model
- [x] State the chosen technique and why in this plan
- [x] Add the optimized-renderer API (new function, stats, runtime tile/schedule/chunk) without changing the baseline
- [x] Implement conservative circle bounding boxes, deterministic ordered candidate-list construction, and tile-parallel rendering
- [x] Add deterministic Part 4 scenes (sparse, heavy overlap, clustered, few, one, huge) in new files
- [x] Build and run the supplied demo/bench on starter and Part 4 scenes; record PASS/FAIL
- [x] Add Part 4 correctness/determinism tests (all scenes, thread counts, tile sizes, edge cases); save the log
- [x] Save a few example images in a clearly named folder
- [x] Write the main benchmark driver (CSV + readable tables, medians, min/max, preprocessing/render split, T1 definition)
- [x] Compare static/dynamic/guided schedules across tile sizes 8/16/32/64 at max threads on clustered + heavy scenes
- [x] Measure per-thread busy time and identify the strongest bottleneck per scene with data
- [x] Record machine info and hyperthreading/turbo relevance
- [x] Produce speedup-vs-threads, tile/schedule, and per-thread busy-time graphs with notes
- [x] Write `part4_notes.md` (work model with measured Nc/t_avg/c_avg, when the optimization loses, static vs dynamic, bottlenecks, measured vs inferred)
- [x] Write `README_part4.md` with exact compile/scene/test/benchmark/graph commands
- [ ] Final verification, `make clean`, remove extra binaries, keep only report artifacts
- [ ] Final summary (technique, files, test status, tables, bottlenecks, work model, manual follow-ups)

## Findings from re-reading the Part 3 baseline

- `render_parallel()` decomposes by rows. For every pixel it executes all `Nc` circle
  tests and blends only the covered circles, so its dominant cost is
  `Nc * width * height` `covers()` calls. The pixel color is accumulated in a local
  `Pixel` and written once, so there are no atomics and no per-blend shared-memory
  round-trips. Circles are applied in original index order.
- Measured baseline at 1024x1024, `clustered`, 500 circles (Part 3 probe):
  524,288,000 `covers()` calls, 224,734 covered pixels (21.4%), 1,077,826 blends
  (0.21% of tests). Per-thread blend counts were 20.2x uneven, but loop times were only
  1.13x uneven because the uniform test loop dominates.
- Baseline scaling at 512x512 `clustered` (Part 3): 189.2990 ms at 1 thread,
  100.9235/54.8234/28.2888 ms at 2/4/8 threads, `Sp = 6.692` at 8 threads. The baseline's
  1-thread time is 1.76x faster than the serial reference because it avoids a shared
  pixel load/store for every blend.
- `render/renderer.cpp` already has the Part 4 stub `render_optimized(width,height,
  circles)` with the rules: preserve order, use `covers()` for the final test, pad any
  bounding box by at least one pixel, and count list-building time inside the function.
- `render/bench.cpp` validates each parallel image against `render_reference()` before
  timing and skips timings on failure. It does not split preprocessing from rendering,
  so Part 4 needs its own driver. The starter does not fix an image size; size is a CLI
  argument. `render/scenes.cpp` has fixed-seed `dense`, `sparse`, `overlap`, `clustered`
  scenes, but the Part 4-specific scenes will live in new files so the starter scenes
  remain untouched.
- Machine (same as Parts 1-3): Intel Xeon E5-2650 v3 @ 2.30 GHz, 10 physical cores /
  20 logical CPUs, 25 MiB L3, 6.9 GiB RAM, Ubuntu 26.04.1 LTS (kernel 7.0.0-43-generic),
  g++ 15.2.0, libgomp 16, `_OPENMP = 201511` (OpenMP 4.5), runtime advertises OpenMP
  5.1. 8 threads are supported. Hyperthreading is enabled (20 logical / 10 physical), so
  8 workers still fit inside physical cores; turbo and the shared machine can move
  absolute times by a few percent. `matplotlib` is not installed; Pillow 12.1.1 is
  available, so graphs use the same matplotlib-or-Pillow pattern as Parts 1/2. `perf`
  exists but hardware counters were not assumed to be permitted; per-thread
  `omp_get_wtime()` timers are the primary evidence.
- Baseline cost model: `W_base = O(Nc * W * H)` tests plus blends for covered pixels.
  With per-tile culling the expected model is preprocessing `O(Nc * t_avg)` (circle-tile
  overlaps) plus list construction, and rendering `O(W * H * c_avg)` tests, where
  `c_avg` is the average candidate count per tile. The expected test-loop speedup is
  about `Nc / c_avg`, reduced by preprocessing and by blending when many candidates
  actually cover a pixel.

## Decisions

- **Primary technique: Option B, ordered per-tile candidate lists.** For each tile, store
  the circles whose padded bounding box overlaps it, in ascending original index. Rendering
  a tile then tests only its candidates, while still applying them in the exact original
  order. This preserves correctness by construction and avoids the circle-major ordering
  problem entirely. It also gives a natural tile-level parallel task with a tunable
  granularity. A plain per-circle bounding box (Option A) would still force a decision
  about how to traverse pixels in order and would have poor load balance on clustered
  scenes, so Option B is the primary technique.
- **API:** keep `render_parallel()` unchanged. Implement the existing starter entry point
  `render_optimized(width,height,circles)` as the simple wrapper (reads `RENDER_TILE`,
  default 16) and add a new function for the driver:
  `render_parallel_optimized(width,height,circles,tile_size,schedule_kind,chunk_size,stats)`.
  The new declaration is additive; no existing signature or starter behavior changes.
  `render_optimized` goes through the same code path, so `render_demo ... optimized` and
  the supplied `render_bench` keep working.
- **Runtime knobs:** tile size via the new function argument and `RENDER_TILE` for the
  wrapper; schedule kind and chunk via `omp_set_schedule()` plus `schedule(runtime)` in the
  render loop. The driver can therefore compare configurations without recompiling.
  Default optimized configuration: tile 16, static schedule.
- **Conservative bounding box:** use `double` arithmetic
  `x >= w*(cx-r)-0.5` / `x <= w*(cx+r)-0.5`, take `floor(lower)-1` and `ceil(upper)+1`,
  clamp to the image, and skip the circle when the clamped box is empty. The final
  coverage decision is still `covers()` with the reference's `float` pixel-centre test and
  inclusive `<=`. Radius is treated as `|r|` because coverage uses `r*r`.
- **Deterministic ordered construction:** compute all circle boxes in parallel, count
  candidates per tile with per-thread local count arrays, reduce them, compute CSR offsets
  with a serial prefix sum over tiles, then fill by iterating circles in ascending index
  order and appending each circle index to its tiles. Because the fill visits circles in
  ascending order, every tile list is ascending. The prefix/fill serial parts are O(T)
  and O(Nc*t_avg); they are cheap for the measured scenes and their measured fraction is
  reported as a potential Amdahl limit.
- **Tile-parallel rendering:** flatten tiles row-major, render whole tiles with
  `#pragma omp parallel for schedule(runtime)`. Each tile owns disjoint pixels, so no
  locks or atomics are needed. The inner pixel loop walks the CSR candidate list in order,
  calls `covers()`/`blend()` exactly as the reference, and writes each pixel once. Tile
  size stays a parameter, so the sweep can expose the imbalance/dispatch/preprocessing
  tradeoff.
- **Scenes:** put Part 4 scenes in new `render/part4_scenes.{h,cpp}` with fixed seeds:
  `p4_sparse`, `p4_heavy`, `p4_corner`, `p4_few`, `p4_one`, `p4_huge`. The starter scenes
  are still accepted by the drivers and are used for compatibility checks. All performance
  measurements use the same 1024x1024 image size unless stated otherwise.
- **Timing definition:** for each scene, `T1` is the **baseline** `render_parallel` at one
  thread and is used for every `Sp = T1/Tp` in that scene. Each version's self-relative
  speedup (`T1_version / Tp_version`) is also reported. Optimized totals are measured
  externally with `omp_get_wtime()`; preprocessing (including image allocation/white
  fill) and tile rendering are measured inside the optimized function and reported
  separately. One warm-up plus at least 5 timed runs; median, min and max.
- **Output files:** `part4_results.csv` + `part4_bench.txt` for the main table,
  `part4_sweep.csv` + `part4_sweep.txt` + `part4_busy.csv` for the schedule/tile/busy
  study, `part4_tests.log` for correctness, `part4_images/` for report images, and root
  PNGs for graphs. No file output happens inside a timed region.

## Findings / Results

- Reading and cost model: complete (above).
- Implementation: `render_parallel_optimized()` added in `render/renderer.cpp` with an
  additive declaration in `renderer.h`; `render_optimized()` is the starter-facing
  wrapper (`RENDER_TILE`, schedule left to `OMP_SCHEDULE`). Conservative padded boxes,
  per-thread candidate counting, serial prefix sum, ascending-index fill, and tile-parallel
  rendering are implemented. `render_parallel()` is unchanged.
- Build: `make render_demo render_bench` succeeds with no warnings. Quick demo checks:
  `dense` 256x256, `sparse` 257x257 and `clustered` 256x256 all report
  `correctness=PASS` with the optimized variant. The supplied bench also passes all four
  starter scenes at 1/2/4/8 threads (it reports both baseline and optimized rows).
- Part 4 scenes: `render/part4_scenes.{h,cpp}` add fixed-seed `p4_sparse` (2000 tiny),
  `p4_heavy` (400 large), `p4_corner` (700 packed top-left + 60 scattered), `p4_few` (8),
  `p4_one` (1 huge), `p4_huge` (32 huge). Starter scenes remain available through
  `make_any_scene()` and `all_scene_names()`.
- Correctness: `render/part4_tests.cpp` (run by `render/run_part4_tests.sh`) reports
  **202 checks, 0 failed -> PASS** in `part4_tests.log`. It covers all 10 scenes at
  257x193 (non-divisible by 16), tile sizes 1/2/3/7/8/16/32/64/128/512, static/dynamic/
  guided schedules, 1/2/4/8 threads, zero/one-pixel images, zero circles, partly/fully
  outside circles, exact tile boundaries, zero/negative radius, and 64 concentric circles.
  Every optimized result is within tolerance, **bit-exact** with the reference, and
  identical over repeated renders.
- Tests/benchmarks: `render/part4_tests.cpp` is clean (202/202). The main benchmark
  (`render/bench_part4.cpp`) and the schedule/tile sweep (`render/sweep_part4.cpp`) were
  run at 1024x1024 with 5 timed runs after one warm-up. Every scene is reference-checked
  at every thread count before timing; no scene failed. Main tables:
  `part4_bench_static.txt` (tile 16 static) and `part4_bench_dynamic.txt` (tile 16,
  dynamic chunk 4), with matching `part4_results_*.csv` and `part4_busy_*.csv`.
- Measured self speedup at 8 threads (tile 16, dynamic,4) and baseline-T1-relative
  `Sp`: `p4_sparse` 4.99 / 1560.8, `p4_corner` 5.21 / 530.8, `p4_heavy` 6.77 / 10.6,
  `p4_few` 4.18 / 8.5, `p4_huge` 6.37 / 6.0, `p4_one` 4.05 / 3.1. The last two are
  pathological: `c_avg = Nc`, so there is nothing to cull. `p4_one` is 29% slower than
  the baseline at one thread and 19% slower at eight; `p4_huge` loses 5% at one thread
  and breaks even at eight.
- Schedule study: static block scheduling is catastrophic on the clustered scene
  (`p4_corner` tile 16: 9.72 ms static vs 2.67 ms dynamic,4 at 8 threads; per-thread busy
  max/mean 4.86 and 79.6% idle vs 1.01 and 2.0%). Small tiles can lose on heavy overlap
  because preprocessing scales with candidate entries (tile 8 costs 37.75 ms of
  preprocessing vs 6.87 ms at tile 16). Best measured configurations: `p4_corner`
  tile 8 dynamic,16 at 2.25 ms; `p4_heavy` tile 32 guided,4 at 78.27 ms.
- Work model (`part4_workmodel.txt`): test reduction is 1417x (`p4_sparse`), 547x
  (`p4_corner`), 3.4x (`p4_heavy`), 5.8x (`p4_few`), 1.0x (`p4_one`, `p4_huge`). Blends
  are 7.7%, 25.6%, 78.8%, 71.9%, 100%, 98.8% of the remaining candidate tests
  respectively. This explains both the large wins and the break-even cases.
- Graphs: `part4_speedup_vs_threads.png`, `part4_tile_schedule.png`,
  `part4_busy_threads.png` produced by `render/plot_part4.py` using the Pillow fallback
  (matplotlib absent), with labeled axes and one-line notes.
- Notes/README: `part4_notes.md` has the technique justification, correctness evidence,
  machine info, medians/speedups/efficiencies, schedule/tile table, work model,
  per-scene bottleneck table (measured vs inferred), and follow-ups. `README_part4.md`
  has the exact build/test/benchmark/workmodel/graph commands.
- Example images: `part4_images/p4_{sparse,heavy,corner}_512_{reference,baseline,optimized}.png`
  from `render/run_part4_examples.sh`.

## Next steps

1. [x] Final verification: rebuild, rerun the demo and the Part 4 tests from the committed
   tree, confirm timing files are consistent with the notes.
2. [x] `make clean` + remove `render/.part4_build`, root PPMs and `part4_images/`; keep the
   `part4_*` report artifacts and the code changes.
3. [x] Commit the Part 4 work and produce the final summary. Commit `69100c1`.

---

# Part 5: MPI Collectives and Communication-Cost Reasoning

## Goal

No MPI code is required. Design the MPI data flow for the tiled renderer, derive and
evaluate broadcast costs under `T(m) = alpha + beta*m`, compare ring/2D-mesh/hypercube
broadcast step counts for p = 16, and analyse load balance for a skewed tile-time set on
8 processes. Everything is analytical and lives in a new `part5/` folder; `saxpy/`,
`scan/` and `render/` must not change. `verify_part5.py` is the source of truth for every
number, and two labeled PNG diagrams are produced with a plotting tool.

## Checklist

- [x] Append this Part 5 plan; note available diagram tooling
- [x] A: write the data-flow design (broadcast metadata + circles, tile ownership, gather,
  root assembly), with the Scatterv/Gatherv conditions, a concrete `counts[]`/`displs[]`
  example, the non-contiguity choice, full-circle vs overlap-only recommendation, and
  data-size estimates
- [x] A: draw the data-flow diagram PNG
- [x] B: derive `T_linear` and `T_binomial`, assumptions, p = 8 round-by-round table, and
  the large-message note
- [x] C: compute the p = 8, m = 2^20, alpha = 10 us, beta = 1 ns/B numbers with units
- [x] D: ring / 4x4 mesh / Q4 step counts with assumptions and derivation table
- [x] D: draw the topology PNG with the broadcast steps marked
- [x] E: compute the [2,2,3,3,3,4,5,18] ms load-balance numbers, lower bounds, speedup,
  efficiency and idle time
- [x] E: choose one change, run an LPT simulation for the split 18 ms tile, print the
  assignment, compare compute saved with the extra communication cost
- [x] Write `verify_part5.py`, run it, save `part5_output.txt`
- [x] Write `part5_notes.md` with every formula and intermediate step
- [x] Write `part5/README_part5.md` with the regeneration commands
- [x] Final verification: rerun the script, confirm diagrams, `git status` shows only
  `part5/` and `plan.md`, produce the summary

## Decisions

- **Diagram tooling:** matplotlib and graphviz are not installed (`import matplotlib`
  fails, `dot` is absent); Pillow 12.1.1 is available, so both diagrams are drawn with
  Pillow using explicit geometry (as in Parts 1-4). This is recorded here per the task
  rules.
- All statements are **model-based**; nothing is measured. Assumptions are stated next to
  each formula (single-port, contention-free links, no pipelining, store-and-forward
  step counting, tile cost divides evenly among sub-tiles).
- `verify_part5.py` recomputes and asserts every number and prints a structured report;
  `part5_output.txt` is its captured output.
- Recommended load-balancing change: over-decompose the 18 ms tile into four 4.5 ms
  pieces and assign with LPT (largest processing time first), then verify the makespan by
  simulation.
- Data-size assumptions: circle record = 7 float32 fields = 28 bytes; final image RGBA
  float32 = 16 bytes/pixel (16 MiB at 1024x1024); 16x16 tile = 4096 bytes; the task-E
  tile payload is 32x32 RGBA float32 = 16,384 bytes.

## Findings / Results

- Plan appended; tooling checked: matplotlib and graphviz are absent, Pillow 12.1.1 is
  available (diagrams will use Pillow).
- `part5/verify_part5.py` written and run: 57 numeric checks, all OK; output saved to
  `part5/part5_output.txt`.
- A (sizes): circle record 28 B (11,200 B at Nc=400; 56,000 B at Nc=2000); 1024x1024 RGBA
  float32 image = 16,777,216 B (16 MiB); 16x16 tile = 4,096 B; 4,096 tiles at p=8 = 512
  per rank; 32x32 tile = 16,384 B. Scatterv example 1050x1050/16 -> 66x66 = 4,356 tiles:
  `counts=[545,545,545,545,544,544,544,544]`,
  `displs=[0,545,1090,1635,2180,2724,3268,3812]`; edge tiles are 10 px wide/tall.
- Diagrams: `part5/part5_dataflow.png` (broadcast -> render -> Gatherv -> reassemble, with
  the Scatterv/Gatherv example) and `part5/part5_topologies.png` (ring, 4x4 mesh, Q4 with
  per-node steps and color-coded rounds). Both drawn with Pillow
  (`part5/draw_part5_diagrams.py`); matplotlib and graphviz are absent.
- B/C: `T(msg)=1058.576 us`; linear (7 messages) = 7,410.032 us = 7.410032 ms;
  binomial (3 rounds) = 3,175.728 us = 3.175728 ms; ratio 7/3 = 2.3333; binomial saves
  4,234.304 us, of which only 40 us (0.94%) is latency and 4,194.304 us is bandwidth.
  `beta*m` is 99.06% of one message, so this case is bandwidth-dominated.
- D: ring unidirectional 15 steps, bidirectional 8; 4x4 mesh root at corner 6, at centre 4;
  4x4 torus 4; Q4 hypercube 4. With `T(msg)=1058.576 us`: ring bidi 8,468.608 us, mesh
  corner 6,351.456 us, torus/Q4 4,234.304 us.
- E: total 40 ms, average bound 5 ms, indivisible bound max(5,18)=18 ms; one tile per rank
  makespan 18 ms, speedup 2.2222x, efficiency 27.7778%, idle 104 process-ms. Splitting the
  18 ms tile into four 4.5 ms pieces and LPT gives loads `[5,4.5,4.5,4.5,4.5,6,6,5]`,
  makespan 6 ms, speedup 6.6667x, efficiency 83.3333%, 12 ms saved; the 6 ms makespan is
  optimal for this split (a sub-6 ms packing is impossible). Per-tile queue cost for
  11 tiles (32x32 RGBA, 16 KiB) = 400.4 us, so net gain ~11.60 ms; one batch Gatherv costs
  ~210.224 us; break-even 329.67 per-tile messages.
- `part5/part5_notes.md` and `part5/README_part5.md` written. The notes contain the full
  A-E answers, all intermediate formulas, assumptions, the measured-vs-model distinction
  (nothing measured) and the recommended change. The README has the exact regeneration
  commands and output file names.

## Next steps

1. [x] Final verification: rerun `verify_part5.py`, confirm both PNGs, check `git status`
   shows only `part5/` and `plan.md`, and produce the final summary.
2. [ ] Manual follow-up (student): paste the diagrams into the PDF report and rewrite the
   calculations in your own words; the analysis files are ready and no code changes are
   needed.

