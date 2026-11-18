# Part 4 notes: optimized renderer and performance investigation

All numbers in this file were measured on the machine described below with the
programs in `render/`. The raw data is in `part4_results_static.csv`,
`part4_results_dynamic.csv`, `part4_busy_static.csv`, `part4_busy_dynamic.csv`,
`part4_sweep.csv`, `part4_sweep_busy.csv`, `part4_workmodel.txt`,
`part4_bench_static.txt`, `part4_bench_dynamic.txt`, `part4_sweep.txt`, and
`part4_tests.log`. Graphs: `part4_speedup_vs_threads.png`,
`part4_tile_schedule.png`, `part4_busy_threads.png`. Images:
`part4_images/`.

## 1. Technique and why

**Chosen technique: Option B, ordered per-tile candidate lists (CSR-style).**

For a 1024x1024 image the baseline `render_parallel()` executes
`Nc * width * height` coverage tests. Most circles cannot possibly touch most
pixels, so the tests are wasted. The optimized renderer splits the image into
square tiles, records for every tile the circles whose *padded* bounding box
overlaps that tile, and then renders each tile testing only its candidates.

Why Option B rather than a per-circle bounding box (Option A):

- A per-circle box still needs a decision about how to visit pixels while
  preserving the original order. A circle-major traversal breaks the blend order
  (blending is not commutative), and a pixel-major traversal with per-circle
  boxes is exactly the baseline loop plus a box test.
- Tile-level candidate lists preserve order **by construction**: the list for a
  tile is filled by visiting circles `0..Nc-1`, so it is always ascending. The
  pixel loop then applies `covers()`/`blend()` in exactly the reference order.
- Tiles give a natural parallel task with tunable granularity, which is what lets
  the study separate scheduler imbalance from preprocessing cost.
- The lists are built once per render and shared read-only by all workers.

The bounding box is only a filter. The final coverage test is the reference
`covers()` and the blend is the reference `blend()`; no approximation is made.

## 2. Implementation

Files:

- `render/renderer.h`: additive declarations `RenderSchedule`, `RenderStats`,
  `render_parallel_optimized(...)`. Existing declarations are unchanged.
- `render/renderer.cpp`: unchanged `render_parallel()` (frozen Part 3 baseline),
  new optimized path, and `render_optimized()` as the starter-facing wrapper.
- `render/part4_scenes.{h,cpp}`: fixed-seed Part 4 scenes, plus
  `make_any_scene()`/`all_scene_names()` so the drivers also accept the starter
  scenes.
- `render/part4_tests.cpp`, `render/part4_workmodel.cpp`,
  `render/bench_part4.cpp`, `render/sweep_part4.cpp`,
  `render/part4_examples.cpp`, `render/plot_part4.py`, and the
  `render/run_part4_*.sh` scripts. The top-level `Makefile` is untouched;
  everything Part 4-specific is built by its own script.

Configuration is runtime-only: tile size, schedule kind (static/dynamic/guided/
auto/runtime) and chunk size are arguments to
`render_parallel_optimized()`. `render_optimized()` reads `RENDER_TILE`
(default 16) and leaves the schedule to `OMP_SCHEDULE`, so the supplied
`render_demo`/`render_bench` keep working. No implementation calls
`omp_set_num_threads()`.

Pipeline inside one optimized call:

1. **Boxes** (parallel over circles): conservative padded pixel box. The covered
   x range is `w*(cx-r)-0.5 .. w*(cx+r)-0.5`; the code takes `floor(lower)-1`
   and `ceil(upper)+1`, clamps to the image, and discards empty boxes. Radius is
   `|r|` because coverage uses `r*r`; non-finite coordinates produce an empty
   box.
2. **Counting** (parallel): per-thread local count arrays over tiles, summed
   deterministically into `counts[t]`.
3. **Prefix sum** (serial, O(tiles)): `offsets[t]` is the start of tile `t` in
   the flat candidate array.
4. **Fill** (serial, O(entries)): visit circles in ascending index order and
   append the circle index to every tile its box overlaps. This is what
   guarantees each tile's slice is sorted ascending.
5. **Render** (parallel over flattened tiles): each tile owns disjoint pixels,
   so no atomics or locks. Inner loop walks the tile's CSR slice in order and
   calls `covers()`/`blend()`, writing each pixel once.

Preprocessing includes image allocation and the white fill, so
`preprocess_ms + render_ms` is the whole renderer cost. Both are measured with
`omp_get_wtime()`; the benchmark also measures the total externally and checks
the split sums to it. No timing region writes files.

## 3. Scenes

Fixed `Lcg` seeds, 1024x1024 unless stated. `Nc` = circles.

| scene | Nc | purpose |
|---|---:|---|
| `p4_sparse` | 2000 | tiny circles (r 0.002–0.006), uniform; most pixels empty |
| `p4_heavy` | 400 | large circles (r 0.18–0.40), alpha 0.05–0.18; heavy overlap |
| `p4_corner` | 760 | 700 packed near (0.18,0.18) plus 60 scattered; uneven spatial load |
| `p4_few` | 8 | medium circles; low circle count, culling has little to win |
| `p4_one` | 1 | one huge circle; **no culling possible** (pathological) |
| `p4_huge` | 32 | huge circles covering everything; **no culling possible** |

Starter scenes `dense`, `sparse`, `overlap`, `clustered` are also accepted by
every Part 4 driver and are part of the correctness suite.

## 4. Correctness

`render/run_part4_tests.sh` builds and runs `render/part4_tests.cpp`; output is
`part4_tests.log`: **202 checks, 0 failed -> PASS**.

- All 10 scenes (6 Part 4 + 4 starter) at 257x193 (size not divisible by 16),
  1/2/4/8 threads: optimized is within `images_close` tolerance, **bit-exact**
  with `render_reference()`, and identical across three repeated renders. The
  baseline is checked at the same time.
- Tile-size sweep 1/2/3/7/8/16/32/64/128/512 on `p4_corner` and `p4_heavy`.
- Static, dynamic chunk 1 and 4, guided chunk 4.
- Edge cases: zero circles (33x17, 0x0, 0x7, 7x0), one-pixel image, fully
  outside circles, partly outside circles, exact tile-boundary coverage (a
  circle whose covered pixel centre lies exactly on a 16-pixel tile boundary),
  zero radius, negative radius, and 64 concentric translucent circles (order
  trap).

Per-scene status at 1/2/4/8 threads and all tested tile sizes: PASS for every
scene, including the two pathological scenes (they are correct, just not
faster).

The work-model tool reported candidate entries and blends for every scene, and
they match the numbers in the renderer's stats, so the lists are not silently
dropping candidates.

## 5. Machine and method

- Intel Xeon E5-2650 v3 @ 2.30 GHz, 10 physical cores / 20 logical CPUs
  (hyperthreading on), 25 MiB L3, 6.9 GiB RAM.
- Ubuntu 26.04.1 LTS, kernel 7.0.0-43-generic, g++ 15.2.0, libgomp 16,
  `_OPENMP = 201511` (OpenMP 4.5; the runtime advertises 5.1).
- 8 threads stay inside the 10 physical cores, so the 8-thread rows are not
  SMT experiments. No core pinning, turbo and other load on the shared machine
  move absolute times by a few percent; the benchmark reports medians of 5 runs
  after one warm-up, plus min/max.
- `T1` is the **baseline** `render_parallel()` on 1 thread for that scene, used
  for every `Sp = T1/Tp`. `self` is the version's own `T1/Tp`; `eff = Sp/p`.
- Graphs use the Pillow fallback because matplotlib is not installed. `perf`
  exists but hardware counters were not assumed to be permitted, so the
  per-thread evidence is `omp_get_wtime()` timers around each worker's tiles.

The primary table below uses the **tuned** configuration (tile 16, dynamic
chunk 4). The default wrapper is tile 16 static; static is included in the
schedule study and is pathological on spatially clustered scenes. Both full
tables are in the CSV/bench files. Reproduce with:

```
./render/run_part4_bench.sh 1024 5 1,2,4,8 p4_sparse,p4_heavy,p4_corner,p4_few,p4_one,p4_huge 16 dynamic 4
```

## 6. Main results (tile 16, dynamic chunk 4, medians)

`base` is the baseline median in ms, `opt` the optimized median, `Sp` is
baseline-T1 relative, `self` is optimized-T1 relative, `Ep = Sp/p`. `pre` and
`render` are the optimized internal medians at that thread count (ms).

### p4_sparse (2000 tiny circles)

| p | base | opt | Sp | self | Ep | pre | render |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3221.58 | 10.29 | 313.0 | 1.00 | 313.02 | 2.07 | 8.00 |
| 2 | 1674.57 | 7.56 | 426.3 | 1.36 | 213.13 | 2.02 | 5.54 |
| 4 | 942.30 | 3.31 | 974.6 | 3.11 | 243.65 | 0.95 | 2.31 |
| 8 | 490.55 | 2.06 | **1560.8** | 4.99 | 195.10 | 0.97 | 1.10 |

Candidate tests are cut 1417x (2.097e9 -> 1.480e6). The p=1 preprocess value is
an outlier from first-call page faults; the static table shows 0.83 ms. The
render at p=1 is ~400x faster than the baseline total, not 1417x, because a
1.05M-pixel image write plus allocation is a fixed floor.

### p4_heavy (400 large, overlapping)

| p | base | opt | Sp | self | Ep | pre | render |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 916.24 | 584.01 | 1.62 | 1.00 | 1.57 | 6.79 | 577.49 |
| 2 | 480.02 | 292.36 | 3.13 | 2.00 | 1.57 | 6.70 | 285.48 |
| 4 | 257.51 | 155.07 | 5.91 | 3.77 | 1.48 | 7.07 | 148.04 |
| 8 | 142.01 | 86.31 | **10.62** | 6.77 | 1.33 | 6.82 | 79.21 |

Only 3.4x fewer candidate tests, because 78.8% of the remaining candidate tests
actually cover a pixel. Culling cannot remove blends, so render speedup vs the
baseline is ~1.6x at one thread. Preprocessing is ~1.2% of the p=1 total; the
Amdahl ceiling from that fraction is 7.2x and the measured self speedup is
6.77x (94% of the ceiling).

### p4_corner (unevenly clustered)

| p | base | opt | Sp | self | Ep | pre | render |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1197.90 | 11.75 | 101.9 | 1.00 | 101.91 | 0.89 | 10.83 |
| 2 | 642.46 | 5.73 | 209.1 | 2.05 | 104.57 | 0.85 | 4.83 |
| 4 | 326.10 | 3.92 | 305.2 | 3.00 | 76.31 | 1.05 | 2.86 |
| 8 | 189.57 | 2.26 | **530.8** | 5.21 | 66.34 | 0.90 | 1.36 |

At p=8 the preprocessing (0.90 ms) is 40% of the 2.26 ms total, so the Amdahl
ceiling is 6.06x; the measured self speedup is 5.21x. With static scheduling the
same configuration gets only 1.03x self speedup (see below), which is entirely
scheduler imbalance, not culling.

### p4_few (8 medium circles)

| p | base | opt | Sp | self | Ep | pre | render |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.20 | 9.87 | 2.05 | 1.00 | 2.05 | 0.80 | 9.11 |
| 2 | 14.02 | 6.46 | 3.13 | 1.53 | 1.56 | 0.95 | 5.41 |
| 4 | 6.92 | 3.68 | 5.49 | 2.68 | 1.37 | 0.91 | 2.74 |
| 8 | 3.83 | 2.36 | **8.55** | 4.18 | 1.07 | 0.92 | 1.44 |

Only 5.8x fewer tests and 71.9% of them blend, so the win is modest and the
fixed 1M-pixel image write is a large share of 2.36 ms.

### p4_one (1 huge circle) — pathological

| p | base | opt | Sp | self | Ep | pre | render |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.15 | 9.22 | 0.78 | 1.00 | 0.78 | 0.75 | 8.41 |
| 2 | 5.43 | 5.97 | 1.20 | 1.55 | 0.60 | 0.92 | 5.09 |
| 4 | 2.77 | 3.51 | 2.02 | 2.62 | 0.51 | 0.92 | 2.61 |
| 8 | 1.92 | 2.28 | 3.14 | 4.05 | 0.39 | 0.93 | 1.34 |

`c_avg = Nc = 1`: every tile lists the only circle, so candidate tests are
identical to the baseline and the indirect list index plus preprocessing are
pure overhead. The optimized renderer is 29% slower at one thread and 19%
slower at eight.

### p4_huge (32 huge circles) — pathological

| p | base | opt | Sp | self | Ep | pre | render |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 168.88 | 178.00 | 0.95 | 1.00 | 0.95 | 2.90 | 174.98 |
| 2 | 89.47 | 99.89 | 1.69 | 1.78 | 0.85 | 3.04 | 96.93 |
| 4 | 53.83 | 51.63 | 3.27 | 3.45 | 0.82 | 2.61 | 48.81 |
| 8 | 27.74 | 27.96 | 5.99 | 6.37 | 0.75 | 2.53 | 25.43 |

`c_avg = 32 = Nc` and 98.8% of candidate tests blend. The lists cannot filter
anything, so the result is a break-even at 8 threads (0.99x vs baseline T1) and
a 5% loss at one thread. This is the honest "optimization loses or breaks even"
case alongside `p4_one`.

## 7. Schedule and tile-size study (8 threads, 1024x1024)

Median total ms, 5 runs. Preprocessing in parentheses is the tile's
`pre_ms`; `c_avg` is the mean candidate count per tile.

### p4_corner

| tile | static | dynamic,1 | dynamic,4 | dynamic,16 | guided,4 | c_avg | pre (static) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 7.93 | 3.16 | 2.40 | **2.25** | 7.61 | 0.93 | 1.02 |
| 16 | 9.72 | 3.65 | 2.67 | 2.97 | 8.33 | 1.39 | 1.94 |
| 32 | 10.29 | 3.67 | 5.72 | 4.82 | 13.28 | 2.64 | 1.01 |
| 64 | 19.12 | 11.08 | 11.28 | 19.17 | 33.20 | 6.26 | 1.71 |

### p4_heavy

| tile | static | dynamic,1 | dynamic,4 | dynamic,16 | guided,4 | c_avg | pre (static) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 151.43 | 117.16 | 117.60 | 117.93 | 115.78 | 116.2 | 37.75 |
| 16 | 105.14 | 84.37 | 83.53 | 84.23 | 83.62 | 118.6 | 6.87 |
| 32 | 111.37 | 79.86 | 80.16 | 82.21 | **78.27** | 123.6 | 2.48 |
| 64 | 105.25 | 81.69 | 84.13 | 87.31 | 87.26 | 133.8 | 1.71 |

Findings:

- **Static block scheduling is the dominant problem on clustered work.** With
  tiles flattened row-major, thread 0 gets the top rows, which is where
  `p4_corner` is packed. Per-thread busy time at tile 16 shows one worker doing
  12.52 ms while the others do 0.81–2.08 ms (max/mean 4.86, idle 79.6%). With
  dynamic chunk 4 the same tile runs 2.86 ms at max/mean 1.01 and 2.0% idle,
  and the median total drops 9.72 -> 2.67 ms. This is a 3.6x scheduler effect
  that has nothing to do with the culling technique.
- **Small tiles can lose on heavy overlap** because preprocessing is
  proportional to candidate entries: tile 8 produces 1.90M entries for
  `p4_heavy` and costs 37.75 ms of preprocessing, more than the 6.87 ms at tile
  16 or 2.48 ms at tile 32. The render improves only from 98.1 to 113.6 ms going
  from tile 16 to 8 (static pair), so tile 8 is a net loss there. The best
  `p4_heavy` configuration is tile 32 guided,4 at 78.27 ms; all dynamic/guided
  variants at tile 16–32 are within ~7%.
- **Guided is not automatically better.** It behaves like static when its chunk
  is large relative to the remaining work (tile 64), and on `p4_corner` tile 8
  it is close to static despite being dynamic in name, because guided hands out
  large initial chunks that land on the dense corner.
- **`p4_heavy` static imbalance is milder** (max/mean 1.33, 24.8% idle) because
  candidates are spread over the image, but it is still 105.1 vs 83.5 ms with
  dynamic chunk 4.

Guidance for this code: use dynamic with a chunk that is a few tiles (4–16) at
tile 8–16, or guided only when the work per tile is large and uniformly spread.
The default `RENDER_TILE=16` plus `OMP_SCHEDULE` is therefore not enough on its
own for clustered scenes; the tuned path should set `dynamic,4`.

## 8. Work model

From `part4_workmodel.txt` (1024x1024, tile 16). Baseline tests are
`W*H*Nc`; candidate tests are `W*H*c_avg`; `test_red` is their ratio; `blends`
is the number of circle/pixel pairs that actually pass `covers()`.

| scene | Nc | tiles | entries | c_avg | t_avg | base tests | cand tests | test_red | blends | blends/tests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p4_sparse | 2000 | 4096 | 5781 | 1.411 | 2.890 | 2.097e9 | 1.480e6 | 1417.1x | 113465 | 7.7% |
| p4_heavy | 400 | 4096 | 485786 | 118.600 | 1214.5 | 4.194e8 | 1.244e8 | 3.4x | 97963005 | 78.8% |
| p4_corner | 760 | 4096 | 5690 | 1.389 | 7.487 | 7.969e8 | 1.457e6 | 547.1x | 373420 | 25.6% |
| p4_few | 8 | 4096 | 5616 | 1.371 | 702.0 | 8.389e6 | 1.438e6 | 5.8x | 1033463 | 71.9% |
| p4_one | 1 | 4096 | 4096 | 1.000 | 4096.0 | 1.049e6 | 1.049e6 | 1.0x | 1048356 | 100.0% |
| p4_huge | 32 | 4096 | 131072 | 32.000 | 4096.0 | 3.355e7 | 3.355e7 | 1.0x | 33162957 | 98.8% |

Interpretation (measured counts; the cost attribution is inference):

- The candidate list predicts the test-loop reduction well for sparse and
  clustered scenes (1417x and 547x fewer tests), but the end-to-end win is
  smaller (313x and 102x at one thread) because a 1024x1024 image write and
  allocation are a fixed ~1–2 ms floor.
- For `p4_heavy` and `p4_few` the limit is blending, not testing: 78.8% and
  71.9% of candidate tests actually blend. The theoretical test reduction is
  3.4x/5.8x, and the measured renderer gain at one thread is 1.6x/2.1x. Adding
  a cheap reject (for example, per-tile candidate sub-boxes) cannot fix this:
  the remaining candidates really do cover the pixel.
- For `p4_one` and `p4_huge`, `c_avg = Nc`, so the list contains every circle
  for every tile and `test_red = 1.0`. The optimized path is the baseline plus
  preprocessing and one extra indirection; it loses or breaks even.
- `t_avg` (tiles per circle) grows with circle radius: 2.9 for tiny
  `p4_sparse` circles up to 4096 (every tile) for the huge ones. Preprocessing
  is `O(Nc*t_avg)` entries plus `O(tiles)` prefix work; at tile 8 the entry
  count grows another 4x, which is why tiny tiles are expensive on heavy
  scenes.

## 9. Bottleneck per scene (measured, plus labeled inference)

| scene | strongest measured bottleneck | data | inference |
|---|---|---|---|
| p4_sparse | fixed per-call cost after culling | pre 0.97 ms vs render 1.10 ms at p=8 (47% pre); test reduction 1417x but render only ~400x faster than baseline at p=1 | image allocation + 12 MB white fill + memory write bandwidth dominate; culling has already removed the arithmetic |
| p4_heavy | blending | 78.8% of 124.4M candidate tests blend; test reduction only 3.4x; render 79.2 ms of 86.3 ms at p=8 | arithmetic and branch cost inside `blend()`, not coverage tests; further wins need a different blend formulation, not more culling |
| p4_corner | scheduler imbalance, then preprocessing | static: 79.6% idle, max/mean 4.86; dynamic: 2.0% idle, 2.67 ms; pre 0.90 ms = 40% of the 2.26 ms tuned total | block-static assignment aligns the dense corner with one worker; after balancing, the serial prefix/fill and the 12 MB image write set the Amdahl ceiling (~6.1x, measured 5.2x) |
| p4_few | fixed overhead + blending | 8.389e6 baseline tests total, 71.9% of candidate tests blend, 2.36 ms total at p=8 | too little work to amortize the 1M-pixel write and preprocessing; scheduling cannot recover it |
| p4_one | no filtering possible | c_avg = 1 = Nc, test_red 1.0x, optimized 9.22 vs baseline 7.15 ms at p=1 | each pixel really needs the one circle; the optimized path adds preprocessing and an indirect index |
| p4_huge | no filtering possible | c_avg = 32 = Nc, 98.8% of tests blend, 27.96 vs 27.74 ms at p=8 | same as p4_one; only per-thread imbalance (max/mean 1.31 static) offers a small gain |

Barrier/wait time is visible as `span - busy` in `part4_busy_*.csv`; the largest
values are the static clustered rows (79.6% of span idle). For dynamic runs the
idle is 0.1–2.8%, so synchronisation is not a bottleneck there.

## 10. What I would do next

- Parallelize the CSR fill and prefix sum (or build the lists once per scene and
  reuse them across frames, which is realistic for animation) to remove the
  preprocessing floor that caps `p4_corner`/`p4_sparse` at high thread counts.
- Keep the ordered-list design; add per-tile candidate count sorting or a
  coarse quadtree only if a scene has both many candidates and mostly-empty
  pixels, which none of the measured scenes does.
- For blend-bound scenes, test a version that accumulates a precomputed
  per-circle premultiplied color, or skips the per-component `1-alpha`
  computation, and compare bit-exactness with the reference before timing.
- Re-run on an idle machine with core pinning and `perf` to separate memory
  bandwidth from branch cost on `p4_sparse`/`p4_heavy`.

## 11. Manual follow-ups

- Delete `part4_images/` and the generated CSVs/PPMs before zipping the
  submission (keep only the notes and README you want to submit).
- `make clean` removes the six starter binaries; the Part 4 binaries live in
  `render/.part4_build/` and must be removed separately.
- The graphs are regenerated by `python3 render/plot_part4.py`; they require
  `part4_results_dynamic.csv`, `part4_sweep.csv` and `part4_sweep_busy.csv`.
