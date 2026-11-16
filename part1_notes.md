# Part 1 Analysis: SAXPY and Parallel Work-Sharing

All numbers below are measured on this machine; nothing is estimated unless it is
explicitly labeled as an assumption or a derived calculation from measured times.

## Measurement setup and definitions

- Machine: Intel Xeon E5-2650 v3 @ 2.30 GHz; 10 physical cores, 20 logical CPUs
  (2 threads/core, one socket); 25 MiB L3; 6.9 GiB RAM; Ubuntu 26.04.1 LTS,
  kernel 7.0.0-43-generic.
- Compiler: g++ 15.2.0 (Ubuntu 15.2.0-16ubuntu1), libgomp 16.
  Compile-time `_OPENMP = 201511` (OpenMP 4.5); the runtime advertises
  `_OPENMP = 202111` (OpenMP 5.1) in `OMP_DISPLAY_ENV=VERBOSE`.
- Build flags: `-O2 -std=c++17 -Wall -Wextra -fopenmp`.
- `T1` is `saxpy_parallel` run with exactly one thread. `Tp` is the same function at
  `p` threads. `Sp = T1/Tp`, `Ep = Sp/p`.
- Each configuration has one untimed warm-up call followed by 9 timed calls; the
  reported time is the median. Timing uses `omp_get_wtime()`.
- The timed region includes creating, zero-initializing, and destroying the output
  `std::vector`, matching the supplied benchmark's stated convention. No file or
  console I/O is inside the timed region.
- `8` threads are supported by this machine, so no requested thread count was skipped.
- The machine was not idle during the measurements (load average ~1.6, 5.7/6.9 GiB
  RAM used, swap in use), so absolute times are specific to this run. The same shape
  was observed in two independent 5-run measurements before the final 9-run run.

## Measured results

| n | p | Tp (ms) | Sp | Ep |
|---:|---:|---:|---:|---:|
| 1e4 | 1 | 0.009908 | 1.000 | 1.000 |
| 1e4 | 2 | 0.009582 | 1.034 | 0.517 |
| 1e4 | 4 | 0.011946 | 0.829 | 0.207 |
| 1e4 | 8 | 0.012415 | 0.798 | 0.100 |
| 1e6 | 1 | 1.587167 | 1.000 | 1.000 |
| 1e6 | 2 | 1.223516 | 1.297 | 0.649 |
| 1e6 | 4 | 1.093735 | 1.451 | 0.363 |
| 1e6 | 8 | 1.079484 | 1.470 | 0.184 |
| 1e7 | 1 | 40.668898 | 1.000 | 1.000 |
| 1e7 | 2 | 40.678585 | 1.000 | 0.500 |
| 1e7 | 4 | 38.110118 | 1.067 | 0.267 |
| 1e7 | 8 | 38.373547 | 1.060 | 0.132 |

Serial reference medians: 0.009174 ms (1e4), 1.552632 ms (1e6), 42.224660 ms (1e7).
Raw data: `saxpy_results.csv`; plot: `saxpy_speedup.png`.

The submitted implementation writes `Z` in one parallel loop, but `std::vector<float>
out(n, 0.0f)` first zero-fills all n elements serially. A separate diagnostic probe
(`saxpy/probe_bandwidth.cpp`, also 9-run medians) separated that cost from the loop:

| n | zero-fill only (ms) | in-place loop 1/2/4/8 threads (ms) | full function 1/2/4/8 threads (ms) |
|---:|---:|---|---|
| 1e4 | 0.0024 | 0.0119 / 0.0077 / 0.0054 / 0.0050 | 0.0134 / 0.0112 / 0.0097 / 0.0128 |
| 1e6 | 0.4882 | 1.0001 / 0.6073 / 0.3408 / 0.1626 | 1.9727 / 1.4532 / 1.2522 / 1.2407 |
| 1e7 | 34.6745 | 13.6602 / 12.0602 / 11.7927 / 11.7518 | 42.1515 / 41.7002 / 40.0753 / 39.9140 |

## Why SAXPY can be memory-bandwidth-limited even with spare cores

SAXPY performs two floating-point operations per element (`multiply` and `add`) but
moves three data streams through memory. In the ideal single-pass model:

- `float` (4 bytes each): read `X` 4 B + read `Y` 4 B + write `Z` 4 B = **12 bytes/element**.
  Arithmetic intensity = 2 FLOP / 12 B = **1/6 = 0.1667 FLOP/byte**.
- `double` (8 bytes each): 8 + 8 + 8 = **24 bytes/element**.
  Arithmetic intensity = 2 FLOP / 24 B = **1/12 = 0.0833 FLOP/byte**.

At such low arithmetic intensity, the achievable FLOP/s is approximately
`intensity x memory bandwidth`; the cores cannot be kept busy unless the memory system
supplies data fast enough. For float, an ideal 10 GB/s stream allows at most
`0.1667 x 10 = 1.67 GFLOP/s`, regardless of how many cores are idle. The submitted
implementation additionally zero-fills `Z` before writing it (4 more bytes per element
for float), which raises the traffic and lowers the effective intensity further; this
is inherent to returning a value-initialized `std::vector<float>` under the fixed
signature.

Measured evidence for the bandwidth limit:

- At `n = 1e7`, the preallocated in-place loop takes 13.66 ms on 1 thread and 11.75 ms
  on 8 threads, only **1.16x** faster. Memory traffic for the ideal 12 B/element model
  is 120 MB, so that corresponds to about **8.8 GB/s** at 1 thread and **10.2 GB/s** at
  8 threads. More cores do not create more DRAM bandwidth.
- At `n = 1e6` the working set is about 12 MB (three 4 MB arrays), which fits in the
  25 MiB L3. The in-place loop there scales from 1.000 ms to 0.163 ms at 8 threads
  (**6.15x**), showing that cores scale when the data is cache-resident and the
  bandwidth limit is relaxed.
- At `n = 1e7` the working set is about 120 MB, far outside L3, and scaling collapses.
  That is the cache-to-DRAM transition expected for a bandwidth-bound kernel.

The full function at `n = 1e7` is even slower than the in-place probe because the
serial zero-fill takes 34.67 ms of the 39.91 ms measured at 8 threads, about **87%**
of the runtime. Using the probe's same-run full-function one-thread time of 42.15 ms,
even eliminating all remaining work after the zero-fill would cap speedup at
`42.15 / 34.67 = 1.22x`; the measured 1.06x is in that range. The preallocated loop
scaling quoted above is therefore not visible in the submitted function: the serial
initialization dominates it.

## A measured case where adding threads reduced efficiency

**n = 1e4 (primary case).** Speedup falls from 1.000 at 1 thread, to 0.829 at 4
threads, to **0.798 at 8 threads**; efficiency falls from 1.000 to 0.207 and 0.100.
Adding threads makes the function slower, not just less efficient. The likely cause is
fixed per-call overhead: 10,000 elements take only about 9-12 microseconds in total,
while an OpenMP parallel region has fork-join/thread-wake-up cost on the order of
microseconds, and the serial allocate-and-zero-fill step cannot be shared. The probe
shows the in-place loop alone still improves from 11.9 to 5.0 microseconds, but the full
function goes from 13.4 to 12.8 microseconds because the serial parts and parallel
start-up consume the gain. The 1e4 result is therefore overhead-dominated, not
compute- or bandwidth-limited.

**n = 1e7 (secondary case).** Speedup is 1.000 at 2 threads and only 1.060 at 8
threads; efficiency falls to 0.500 and 0.132. Here the cause is different: the serial
zero-fill (34.7 ms) plus DRAM bandwidth saturation of the remaining parallel work. Both
cases are measured, but they have different bottlenecks.

## Conclusions supported by the measurements

1. `saxpy_parallel` is correct: all edge-case checks pass at 1/2/4/8 threads, including
   empty input, n=1, n smaller than the thread count, non-multiples, and a 1e7 random
   array. Inputs are not modified and size mismatch throws `std::invalid_argument`.
2. Speedup is workload-size dependent: about 1.5x at n=1e6, about 1.06x at n=1e7, and
   below 1x at n=1e4.
3. The n=1e4 slowdown coincides with a runtime of only ~10 microseconds, where fixed
   fork-join and serial setup costs are comparable to the useful work.
4. At n=1e7 the serial output zero-fill is measured at 34.7 ms, about 87% of the full
   function at 8 threads. The separate preallocated (in-place) loop scales only ~1.16x,
   consistent with DRAM bandwidth saturation.
5. At n=1e6 the 12 MB working set fits in the 25 MiB L3 and the in-place loop scales
   6.15x, unlike the 1e7 case.

## Assumptions and limitations (not directly measured)

- The split between "DRAM bandwidth" and "page-fault/page-zeroing cost" inside the
  measured zero-fill time is not instrumented. The probe measures the combined
  allocate + page-fault + zeroing cost, so the 34.7 ms should not be attributed to
  `memset` bandwidth alone.
- The L3/DRAM explanation for the 1e6 vs 1e7 difference is inferred from the measured
  working-set sizes (12 MB vs 120 MB) and the 25 MiB L3 size; no hardware performance
  counters were used.
- The machine was shared/loaded and showed memory pressure. Absolute times and the
  exact 8-thread points would differ on an idle machine; the qualitative shape was
  reproduced in repeated runs.
- Only float SAXPY was measured. The double arithmetic is analyzed above but not
  benchmarked, because the starter interface uses `std::vector<float>`.
- 8 threads is below the 10 physical cores, so these measurements do not cover
  oversubscription or hyper-thread contention.
