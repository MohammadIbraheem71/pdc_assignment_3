# Part 3 Analysis: Correct Parallel Circle Rendering

All numbers below were measured on this machine unless they are labelled as a derived
calculation or as reasoning. The measurements that support the race/lock discussion are
deliberately kept separate from the measured renderer timings.

## Measurement setup and definitions

- Machine: Intel Xeon E5-2650 v3 @ 2.30 GHz, 10 physical cores / 20 logical CPUs, 25 MiB
  L3, 6.9 GiB RAM, Ubuntu 26.04.1 LTS (kernel 7.0.0-43-generic), g++ 15.2.0, libgomp 16,
  `_OPENMP = 201511` (OpenMP 4.5), runtime advertises OpenMP 5.1. 8 threads are
  supported. The machine is shared/loaded, so absolute times are run-specific.
- Build flags: `-O2 -std=c++17 -Wall -Wextra -fopenmp`.
- Timing sanity check: 512x512 `clustered` scene (500 circles). One untimed warm-up plus
  7 timed runs per configuration; the reported value is the median measured with
  `omp_get_wtime()`. The timed region is the whole function call and contains no I/O.
- `T1` is `render_parallel` on exactly one thread, not the serial reference. `Tp` is the
  same function on `p` threads, `Sp = T1/Tp`, `Ep = Sp/p`. The serial reference is
  reported separately and `vs_serial = serial_ms/Tp`.
- Correctness tolerance: `images_close` requires equal dimensions and every color
  component to differ by at most `0.0001f` (the default argument). The edge tests also
  require bit-exact equality, which is stronger than the assignment needs.

## Implementation

`render_parallel()` decomposes the image by rows:

```cpp
#pragma omp parallel for schedule(runtime)
for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
        Pixel p{1, 1, 1};
        for (const Circle& c : circles)          // original order, 0..N-1
            if (covers(c, x, y, width, height))  // same helper as the reference
                blend(p, c);                     // same float equation
        image.pixels[y * width + x] = p;         // one write per pixel
    }
}
```

Each pixel belongs to exactly one `(x, y)`, hence to exactly one thread. The pixel color
lives in a local variable while the circle loop runs, so there are no atomics, no locks,
and no read-modify-write of shared memory inside the circle loop. Circles are never
sorted or reassigned. `covers()` and `blend()` are the reference helpers, so the pixel
centre `((x+0.5)/w, (y+0.5)/h)`, the `float` arithmetic, and the inclusive `<=` boundary
test are identical. `schedule(runtime)` is used so Part 4 can change the schedule from
outside with `OMP_SCHEDULE`.

## Verification

- `./render_demo` on `dense`, `sparse`, `overlap`, `clustered` at 256x256 and 1/2/4/8
  threads: **16/16 correctness=PASS**.
- `render/edge_tests.cpp`: zero circles, 0x0 and 0x7 images, non-square 37x21 and 200x150
  images, single circles at the centre/corner/edge, mostly-empty scenes, overlapping
  A-then-B and B-then-A, partly and fully outside circles, 64 concentric circles with
  different colors/opacities, and a mixed 120-circle random scene: **56 checks, 0 failed**
  at 1/2/4/8 threads (`part3_tests.log`). Every case is within tolerance, bit-exact
  against the reference, and identical across four re-renders.
- `render/blend_check.cpp`: the hand calculation matches both renderers exactly
  (`part3_blend.txt`).
- `part3_images/` holds 512x512 reference/parallel PNGs for `dense` and `clustered`; the
  reference and parallel files are byte-identical (`md5sum`), which is consistent with the
  bit-exact test result.

## Measured timing sanity check

512x512 `clustered`, 500 circles, 7-run medians (`part3_timing.txt`):

| variant | p | median (ms) | Sp = T1/Tp | Ep | vs_serial |
|---|---:|---:|---:|---:|---:|
| serial reference | 1 | 333.5046 | – | – | 1.000 |
| parallel | 1 | 189.2990 | 1.000 | 1.000 | 1.762 |
| parallel | 2 | 100.9235 | 1.876 | 0.938 | 3.305 |
| parallel | 4 | 54.8234 | 3.453 | 0.863 | 6.083 |
| parallel | 8 | 28.2888 | 6.692 | 0.837 | 11.789 |

A repeat run in the same session gave serial 323.7280 ms and parallel
187.1182 / 96.5131 / 52.3922 / 28.3514 ms (`Sp` = 1 / 1.939 / 3.571 / 6.600), so the
shape is stable even though the absolute times move by a few percent. Scaling is good up
to 8 threads (6.69x speedup, 0.84 efficiency).

The parallel function at **one thread is already 1.76x faster than the serial reference**
on this scene. Both perform the same per-pixel coverage tests and the same blends in the
same order; the structural difference is that the reference applies each circle to the
whole image and does a load/store of the image pixel for every blend, while the row
version accumulates the pixel in a register and stores it once. On `clustered` a covered
pixel receives 4.8 blends on average (1,077,826 blends over 224,734 covered pixels), so
this removes several memory round-trips per covered pixel. No counter experiment was run
to split the 1.76x between that effect and loop/cache effects, so the exact attribution is
an inference; the measured numbers themselves are from the table above.

## (a) Ordered alpha blending

Equation per component: `P_new = alpha*C + (1 - alpha)*P_old`, applied separately to
R, G and B. Starting from white `P0 = (1,1,1)`, with `A = (1,0,0,0.5)` and
`B = (0,0,1,0.5)`:

**A then B**

1. Apply A: R = 0.5·1 + 0.5·1 = 1.0; G = 0.5·0 + 0.5·1 = 0.5; B = 0.5·0 + 0.5·1 = 0.5
   → `(1.0, 0.5, 0.5)`.
2. Apply B: R = 0.5·0 + 0.5·1.0 = 0.5; G = 0.5·0 + 0.5·0.5 = 0.25;
   B = 0.5·1 + 0.5·0.5 = 0.75 → **`(0.5, 0.25, 0.75)`**.

**B then A**

1. Apply B: R = 0.5·0 + 0.5·1 = 0.5; G = 0.5·0 + 0.5·1 = 0.5; B = 0.5·1 + 0.5·1 = 1.0
   → `(0.5, 0.5, 1.0)`.
2. Apply A: R = 0.5·1 + 0.5·0.5 = 0.75; G = 0.5·0 + 0.5·0.5 = 0.25;
   B = 0.5·0 + 0.5·1.0 = 0.5 → **`(0.75, 0.25, 0.5)`**.

`render/blend_check.cpp` prints these steps and reproduces both results through
`render_reference` and `render_parallel` at pixel (50,50) of a 100x100 image:

```
A then B = (0.500000, 0.250000, 0.750000)
B then A = (0.750000, 0.250000, 0.500000)
BLEND CHECK: PASS
```

The two orders differ in R and B. Blending is **not commutative**: the circle applied
later contributes with full weight `alpha`, while the earlier color is attenuated by
`(1 - alpha)`. Concretely, in A-then-B the final red is `0.5·B.r + 0.5·(0.5·A.r + 0.5·1)`,
so A's contribution to red is halved twice; in B-then-A the weights are swapped. That is
why every pixel must see the circles in increasing index order.

## (b) Race condition: lost update on one pixel

Consider the same white pixel and circles A (index 0) and B (index 1), with an incorrect
circle-parallel decomposition in which thread T1 applies A and thread T2 applies B to the
same pixel without synchronization. Each update is a read-modify-write of the pixel and is
not atomic as a whole.

One possible interleaving (both read the same old value, so A's update is lost):

| step | T1 (applies A) | T2 (applies B) | pixel value after the step |
|---:|---|---|---|
| 0 | – | – | `(1.000, 1.000, 1.000)` |
| 1 | read `P_old = (1,1,1)` | – | `(1.000, 1.000, 1.000)` |
| 2 | – | read `P_old = (1,1,1)` | `(1.000, 1.000, 1.000)` |
| 3 | compute A over white: `(1.0, 0.5, 0.5)` | – | `(1.000, 1.000, 1.000)` |
| 4 | – | compute B over its stale white: `(0.5, 0.5, 1.0)` | `(1.000, 1.000, 1.000)` |
| 5 | write `(1.0, 0.5, 0.5)` | – | `(1.000, 0.500, 0.500)` |
| 6 | – | write `(0.5, 0.5, 1.0)` | **`(0.500, 0.500, 1.000)`** |

The final value is exactly "B applied to white". A's effect disappeared, i.e. a lost
update. It equals neither the correct A-then-B `(0.5,0.25,0.75)` nor B-then-A
`(0.75,0.25,0.5)`: G is 0.5 instead of 0.25 because B was computed from the stale white
pixel, not from A's result. If the two writes are swapped in the table, the result is
A only, `(1.0,0.5,0.5)`: the last writer wins. If T2 reads only after T1's write, the same
program produces the correct A-then-B result. So the output depends on timing and is
nondeterministic.

The three channels are separate `float` stores, so the interleaving can also be finer.
T1 computes A = `(1.0,0.5,0.5)` and T2 computes B = `(0.5,0.5,1.0)`; if T1 stores R,
then T2 stores all three, then T1 stores G and B from its old computation, the pixel
becomes `(0.5, 0.5, 0.5)` — a mixture of the two blends that matches neither order. In
C++ two threads writing the same non-atomic object without synchronization is a data
race and undefined behavior; the interleaving above is only one concrete illustration of
what can happen on this machine's memory model.

Even making each component update atomic does not fix this. One blend then consists of
three separate atomic read-modify-write operations, so another thread can interleave
between the R and G updates, and different channels can end up in different effective
orders. More importantly, atomics give no ordering guarantee between the two circles, so
the required A-then-B order is still not enforced. The same read-modify-write race would
appear if the renderer were parallelized over circles.

The submitted implementation avoids the race by decomposing over pixels: exactly one
thread ever touches a given pixel, and that thread applies all circles in index order to a
local variable.

## (c) Circle-parallel vs pixel-parallel decomposition

The ordering constraint is asymmetric:

- **Per pixel:** circles 0..N-1 must be applied in increasing index order, because
  blending is not commutative (section a). Two different orders can give different colors.
- **Across pixels:** pixel values are independent. Nothing that happens to pixel (x,y)
  affects pixel (x',y'). The only shared inputs are the read-only circle array and the
  image dimensions.

Therefore the natural decomposition is over pixels (or rows/tiles of pixels), with each
thread executing the whole circle loop for the pixels it owns. The submitted baseline does
exactly that and needs no synchronization at all.

A **circle-parallel** decomposition (thread k handles circles k, k+p, ...) violates the
ordering constraint as soon as two circles overlap. Two threads can then read and write
the same pixel concurrently, producing the lost update or the mixed-channel result from
section b. Restoring correctness would require processing the circles in index waves with
a barrier after each circle (or group of circles), which reduces to N barriers for N
circles and leaves little parallel work per wave on small images.

A **critical section or one lock per pixel** around each blend is also a poor fix:

- Mutual exclusion only makes each individual blend atomic; it does not make the order
  match the input order. Whichever thread acquires the lock first wins, so the effective
  order of A and B is still timing-dependent. Correctness would need extra sequencing
  (e.g. a barrier or a per-pixel "next circle index" protocol), which is far more
  complicated than the pixel decomposition.
- Even ignoring ordering, the lock is acquired and released once per blend. On the
  `clustered` 1024x1024 scene there are 1,077,826 blends, so that is over a million
  lock/unlock pairs, plus contention wherever circles overlap (the hot spots).
- Locking serializes precisely the work that is shared: pixels covered by many circles.
  Every other thread touching the same pixel must wait, so the parallel section becomes
  effectively serial in the dense regions while paying lock overhead in the sparse ones.
- The same contention argument applies to a critical section, with the additional
  drawback that `omp critical` is a single global lock here, so even pixels that do not
  overlap would serialize against each other.

**Memory and false sharing.** A `Pixel` is three floats (12 bytes), so a 64-byte cache line
holds 5.33 pixels. If adjacent pixels are written by different threads (for example a
parallel loop over x, or interleaved row assignment), two threads write different pixels
on the same line. The hardware must move that line between cores for every write, which is
false sharing: the writes are logically independent but still cause cache-line ping-pong.
The row decomposition avoids this because each thread owns whole contiguous rows (at
1024 pixels/row a row is 12,288 bytes = 192 cache lines), and only the boundary between
two threads' row ranges can share one line. This reasoning is standard cache behaviour; it
was not measured with hardware counters in Part 3.

## (d) Load calculation for 1024x1024

- Pixels: `1024 x 1024 = 1,048,576`.
- With `p = 8` workers and a perfectly even static row split: `1,048,576 / 8 = 131,072`
  pixels per worker, which is exactly `128` rows of `1024` pixels.
- With `p = 4`: `262,144` pixels = `256` rows per worker.

Equal pixel counts do **not** mean equal work:

- In the baseline every pixel tests all 500 circles, so the **test** cost is uniform:
  `1,048,576 x 500 = 524,288,000` `covers()` calls regardless of the scene.
- Only covered pixels do any blending. Coverage and overlap are not uniform: the
  `clustered` scene packs 400 of its 500 circles into two hot spots. Measured at
  1024x1024, only 224,734 pixels (21.4%) are covered at all, and there are 1,077,826
  blends in total, about 4.8 blends per covered pixel.
- Consequently the blend work per thread is very uneven: in the probe, blend counts range
  from 17,955 to 363,211 across the 8 threads — a **20.2x** spread — while every thread
  still owns exactly 131,072 pixels.
- Empty regions are cheap in blends but not free in tests: a pixel with no circle still
  executes all 500 `covers()` tests. That is why the measured time spread is much smaller
  than the blend spread (below).

Measured per-thread probe (`render/load_probe.cpp`, `part3_load.txt`, 1024x1024
`clustered`, static contiguous rows, barrier before each thread's timed loop):

| p | rows/thread | pixels/thread | covered pixels (min–max) | blends (min–max) | loop ms (min–max) | time ratio |
|---:|---:|---:|---|---|---|---:|
| 8 | 128 | 131,072 | 16,242 – 45,169 | 17,955 – 363,211 | 146.150 – 164.587 | 1.13x |
| 4 | 256 | 262,144 | 45,543 – 62,035 | 62,447 – 383,214 | 285.523 – 306.701 | 1.07x |

The blend work is indeed 20x uneven at 8 threads, but the measured loop time is only 1.13x
uneven, and the thread with the most blends is not the slowest one. This is consistent
with the arithmetic above: blends are only `1,077,826 / 524,288,000 = 0.21%` of the
circle tests, so in the unoptimized baseline the uniform `covers()` tests dominate and a
20x difference in the remaining 0.21% cannot produce a 20x time difference. The small
remaining spread (and its run-to-run variation: 1.13–1.32x over repeated probes, with an
apparent outlier on one thread in one run) is likely cache/memory effects and shared
machine noise rather than blend count. This is a measured result plus an interpretation,
not a hardware-counter attribution.

This also explains why load balance is expected to matter much more in Part 4: culling
removes most of the uniform `covers()` tests, leaving the uneven blend work (and the
uneven number of candidate circles per region) as a larger fraction of the runtime. That
expectation is a prediction, not a Part 3 measurement.

Other costs that can still differ between threads, even with equal rows:

- The image is written row by row, so contiguous rows give contiguous memory access. The
  per-thread working set at 1024x1024 is `131,072 x 12 B = 1.5 MiB` of image plus the
  14 KB circle array (500 x 28 B), so the circle array is re-read for every pixel but is
  small enough to stay in cache. Cache counters were not used.
- The white initialization of the whole image is a serial `std::vector` fill before the
  parallel loop, and the render loop then writes every pixel again; that cost is the same
  for all thread counts and is included in every measured time.

## (e) Conclusions supported by measurements, and assumptions

Measured:

1. `render_parallel` is correct on all supplied scenes (16/16 demo runs) and on 56
   edge/determinism checks at 1/2/4/8 threads, including bit-exact equality with the
   reference.
2. Ordered blending is not commutative: A-then-B gives `(0.5,0.25,0.75)` and B-then-A
   gives `(0.75,0.25,0.5)`, both by hand and through the actual renderers.
3. On 512x512 `clustered`, the baseline scales from 189.2990 ms at 1 thread to 28.2888 ms
   at 8 threads (`Sp = 6.692`, `Ep = 0.837`), and the parallel 1-thread time is 1.76x
   faster than the serial reference (189.2990 vs 333.5046 ms).
4. At 1024x1024 `clustered`, static row splitting gives every one of 8 threads exactly
   131,072 pixels but blend counts from 17,955 to 363,211 (20.2x spread); the measured
   per-thread loop times span only 146.150–164.587 ms (1.13x), because uniform circle
   tests are 99.79% of the coverage work in the baseline.

Reasoning / assumptions (not directly measured in Part 3):

- The lost-update and mixed-channel outcomes in section b are a step-by-step model of the
  C++ data race; the submitted renderer does not contain that race, so it could not be
  triggered by running it. The exact values follow from the supplied blend equation.
- The claim that a critical section or per-pixel lock is slow is based on the number of
  blends (1.08 M at 1024x1024) and the serialization/contention structure. No lock-based
  prototype was benchmarked.
- False sharing at cache-line granularity is standard hardware reasoning; no performance
  counters were read.
- The 1.76x serial-vs-1-thread difference is not decomposed experimentally into
  register-accumulation, loop-order, and cache effects; only the total is measured.
- The expectation that Part 4 will be more sensitive to load imbalance is a prediction
  from the measured test/blend ratio, not measured.
- The machine was shared; absolute times vary by a few percent between runs (the repeat
  timing run is quoted above).
