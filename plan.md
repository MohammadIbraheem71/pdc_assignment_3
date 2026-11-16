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
