# Part 4: optimized renderer, correctness suite, benchmarks and graphs

This part adds an optimized parallel renderer selected from a stand-alone
benchmark driver. The Part 3 baseline (`render_parallel()`) and the starter
reference/`covers()`/`blend()`/demo/bench comparison code are untouched. All
Part 4 files are new or additive under `render/`; the top-level `Makefile` is
unchanged.

## Files

| file | role |
|---|---|
| `render/renderer.h` | additive `RenderSchedule`, `RenderStats`, `render_parallel_optimized()` declarations |
| `render/renderer.cpp` | optimized tiled renderer + `render_optimized()` wrapper; baseline untouched |
| `render/part4_scenes.{h,cpp}` | deterministic scenes `p4_sparse`, `p4_heavy`, `p4_corner`, `p4_few`, `p4_one`, `p4_huge` |
| `render/part4_tests.cpp` | 202-check correctness/determinism/edge-case suite |
| `render/part4_workmodel.cpp` | candidate/blend counts for the work model |
| `render/bench_part4.cpp` | main benchmark (CSV + tables, preprocess/render split) |
| `render/sweep_part4.cpp` | tile-size x schedule sweep + per-thread busy diagnostics |
| `render/part4_examples.cpp` | reference/baseline/optimized PPM images for the report |
| `render/plot_part4.py` | speedup, tile/schedule and busy-time graphs |
| `render/run_part4_*.sh` | build+run scripts for the above |

## Build and quick check

The Part 4 programs compile independently, because the top-level `Makefile`
has no wildcard rule:

```bash
# Baseline demo still works; optimized wrapper is selected with 'optimized'.
make render_demo render_bench
OMP_NUM_THREADS=4 ./render_demo 256 dense optimized
OMP_NUM_THREADS=8 ./render_demo 256 clustered optimized

# Starter bench also exercises the optimized variant on the starter scenes.
./render_bench 256 5 1,2,4,8 dense,sparse,overlap,clustered
```

The optimized wrapper reads `RENDER_TILE` (default 16) and leaves the schedule
to `OMP_SCHEDULE`, e.g. `RENDER_TILE=8 OMP_SCHEDULE=dynamic,4, ...`.

## Correctness and determinism

```bash
./render/run_part4_tests.sh
```

Builds `render/.part4_build/part4_tests` and writes `part4_tests.log`. It covers
all six Part 4 and four starter scenes at 257x193 (non-divisible) and
1/2/4/8 threads, tile sizes 1/2/3/7/8/16/32/64/128/512, static/dynamic/guided,
zero/one-pixel images, zero circles, outside/partly-outside circles, exact tile
boundaries, zero/negative radius, and 64 concentric circles. Every optimized
image must be within `images_close` tolerance, **bit-exact** with the sequential
reference, and identical across three repeated renders. Expected last line:

```
RENDER PART 4 TESTS: 202 checks, 0 failed -> PASS
```

## Main benchmark

```bash
# Default: 1024x1024, 5 runs, 1/2/4/8 threads, all Part 4 scenes, tile 16, static.
./render/run_part4_bench.sh

# Tuned configuration used in the report (balanced on clustered scenes).
./render/run_part4_bench.sh 1024 5 1,2,4,8 \
    p4_sparse,p4_heavy,p4_corner,p4_few,p4_one,p4_huge 16 dynamic 4

# Starter scenes instead.
./render/run_part4_bench.sh 1024 5 1,2,4,8 dense,sparse,overlap,clustered 16 dynamic 4
```

Writes `part4_bench.txt` (readable tables), `part4_results.csv` (one row per
scene/variant/threads) and `part4_busy.csv` (per-thread busy/span at the largest
tested thread count). Every scene is compared with the sequential reference for
both renderers at every thread count **before** any timing; a mismatch prints a
`FAIL ... timings skipped` line and exits 1.

Columns: `preprocess_ms` and `render_ms` are internal medians, `median_ms` is
measured externally; `speedup = T1/median` with `T1` = baseline
`render_parallel` on one thread, `self_speedup` = the version's own `T1/Tp`,
`efficiency = speedup/p`.

The generated CSV uses fixed names. To keep both configurations, run one and
rename, as was done for the report:

```bash
./render/run_part4_bench.sh 1024 5 1,2,4,8 p4_sparse,p4_heavy,p4_corner,p4_few,p4_one,p4_huge 16 static 0
mv part4_bench.txt part4_bench_static.txt
mv part4_results.csv part4_results_static.csv
mv part4_busy.csv part4_busy_static.csv
./render/run_part4_bench.sh 1024 5 1,2,4,8 p4_sparse,p4_heavy,p4_corner,p4_few,p4_one,p4_huge 16 dynamic 4
mv part4_bench.txt part4_bench_dynamic.txt
mv part4_results.csv part4_results_dynamic.csv
mv part4_busy.csv part4_busy_dynamic.csv
```

## Schedule and tile sweep

```bash
./render/run_part4_sweep.sh            # 1024, 5 runs, 8 threads, p4_corner,p4_heavy
./render/run_part4_sweep.sh 1024 5 8 p4_corner,p4_heavy
```

Sweeps tile sizes 8/16/32/64 with static, dynamic chunks 1/4/16 and guided chunk
4. Writes `part4_sweep.txt`, `part4_sweep.csv` and `part4_sweep_busy.csv`. Each
configuration is validated before timing. The busy diagnostics re-run tile 16
static and dynamic,4 with per-worker `omp_get_wtime()` timers.

## Work model

```bash
g++ -O2 -std=c++17 -Wall -Wextra -fopenmp \
    render/part4_workmodel.cpp render/part4_scenes.cpp render/scenes.cpp render/renderer.cpp \
    -o render/.part4_build/part4_workmodel
OMP_NUM_THREADS=8 ./render/.part4_build/part4_workmodel 1024 16 | tee part4_workmodel.txt
```

Reports circles, tiles, candidate entries, `c_avg`, `t_avg`, baseline tests,
candidate tests, test reduction and actual blends per scene.

## Example images

```bash
./render/run_part4_examples.sh
```

Writes `part4_images/<scene>_512_{reference,baseline,optimized}.png` for
`p4_sparse`, `p4_heavy`, `p4_corner` (lossless conversion of the starter text
PPM output; requires Pillow, otherwise PPMs are kept). Delete `part4_images/`
before zipping the submission.

## Graphs

```bash
python3 render/plot_part4.py
```

Requires `part4_results_dynamic.csv`, `part4_sweep.csv`, `part4_sweep_busy.csv`.
Uses matplotlib when installed, otherwise Pillow. Outputs:

- `part4_speedup_vs_threads.png` — baseline vs optimized speedup, six panels.
- `part4_tile_schedule.png` — median runtime vs tile size and schedule.
- `part4_busy_threads.png` — per-worker busy time, static vs dynamic.

## Cleanup

```bash
make clean                      # six starter binaries and render_*.ppm
rm -rf render/.part4_build      # Part 4 binaries
rm -rf part4_images             # report images
```

Keep the `part4_*` text/CSV logs and PNG graphs if they are part of the
submission; they are not produced by `make` and are not removed by
`make clean`.

## Results summary

The optimized renderer is bit-exact with the reference on every scene and tile
size. At 1024x1024, tile 16, dynamic chunk 4, 8 threads, baseline-T1-relative
speedups are 1560.8x (`p4_sparse`), 530.8x (`p4_corner`), 10.6x (`p4_heavy`),
8.5x (`p4_few`), 6.0x (`p4_huge`) and 3.1x (`p4_one`). The last two cannot be
optimized by culling: every tile contains every circle, so the lists add
preprocessing and an indirection. The full analysis, work model and per-scene
bottlenecks are in `part4_notes.md`.
