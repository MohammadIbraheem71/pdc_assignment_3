# Part 2 Analysis: Exclusive Scan and Repeated-Index Compaction

All numbers below were measured on this machine unless they are explicitly labelled as a
derived calculation (arithmetic on measured values or on the code's access pattern) or as
an assumption.

## Measurement setup and definitions

- Machine: Intel Xeon E5-2650 v3 @ 2.30 GHz; 10 physical cores, 20 logical CPUs
  (2 threads/core, one socket); 25 MiB L3; 6.9 GiB RAM; Ubuntu 26.04.1 LTS,
  kernel 7.0.0-43-generic.
- Compiler: g++ 15.2.0, libgomp 16. Compile-time `_OPENMP = 201511` (OpenMP 4.5); the
  runtime advertises OpenMP 5.1.
- Build flags: `-O2 -std=c++17 -Wall -Wextra -fopenmp`.
- **T1 is the parallel function on exactly one thread**, not the serial reference. `Tp` is
  the same parallel function on `p` threads. `Sp = T1/Tp`, `Ep = Sp/p`. The serial
  reference is reported as a separate `serial` row, and `vs_serial = serial_ms/Tp`.
  Reporting both makes the comparison honest: `Sp` measures how well the parallel code
  scales, `vs_serial` measures whether it is actually worth using.
- Each configuration has one untimed warm-up call followed by **9 timed calls**; the
  reported value is the median. Timing uses `omp_get_wtime()`. The timed region contains
  the whole function call (including its internal `std::vector` allocations and result
  construction) and no file or console I/O.
- `8` threads are supported, so no requested thread count was skipped. The machine was
  not idle (load average ~1.6, memory pressure), so absolute times are specific to this
  run; the shape was stable across repeated runs.
- Inputs for the benchmark are pseudo-random integers in `[0, 3]` from the supplied
  fixed-seed `Lcg(12345)`; about one position in four is a repeat.
- Raw data: `scan_results.csv`; readable table: `scan_bench.txt`; plot:
  `scan_speedup.png`; test log: `part2_tests.log`; staged scan: `part2_stages.txt`.

## What was implemented

`exclusive_scan_parallel` is a Blelloch tree scan:

1. Pad the input to `N = next_power_of_two(n)` with the identity element `0` and copy it
   into a heap array of `N` ints (done in one parallel loop).
2. Up-sweep: for `d = 1, 2, 4, ...` do `tree[k + 2d - 1] += tree[k + d - 1]` for
   `k = 0, 2d, 4d, ...`.
3. Set the root `tree[N-1] = 0` (exclusive scan).
4. Down-sweep: for `d = N/2, N/4, ..., 1` exchange the left child with the parent and add
   the old left child into the right child.
5. Return `std::vector<int>(tree, tree + n)`, truncating the padding.

The whole algorithm runs inside **one** `#pragma omp parallel` region with one `omp for`
per level, so the implicit barrier at the end of each `omp for` synchronises the levels
and no fork/join happens per level. The root write uses `omp single` (with its implicit
barrier). Every level writes disjoint indices, so no atomics are needed. The input vector
is only read.

`find_repeats_parallel` follows the required flags → scan → scatter structure:

1. `flags[i] = (input[i] == input[i+1])` for `i` in `[0, n-2]` (so `input[i+1]` is never
   out of bounds), in parallel.
2. `offsets = exclusive_scan_parallel(flags)`.
3. `total = offsets.back() + flags.back()` gives the exact output size.
4. In parallel, `if (flags[i]) result[offsets[i]] = i`.

Because `offsets` increases by one at every flagged position, no two iterations write the
same slot, and ascending `i` produces ascending output indices. `n = 0` and `n = 1`
return an empty result before any scan.

## Verification

- `make scan_demo` and `./scan_demo` for n = 0, 1, 7, 8, 9, 1024, 1025 at
  `OMP_NUM_THREADS = 1, 2, 4, 8`: **28/28 runs PASS** (`scan=PASS repeats=PASS`).
- `scan/edge_tests.cpp` (run by `scan/run_tests.sh`) adds the header examples, all-equal,
  all-distinct, alternating 0/1, alternating pairs, and fixed-seed random arrays of 2, 3,
  17, 1025, 1e5, 1e6 and 1e7 elements, at 1/2/4/8 threads, and checks that the input is
  unchanged: **84 checks, 0 failed** (`part2_tests.log`).
- `scan/scan_stages.cpp` reproduces the staged n=16 walkthrough below and its final array
  equals the expected exclusive scan exactly (`STAGED SCAN: PASS`).

## Staged n = 16 tree scan

Input `[1,2,3,4]` repeated four times. The root is set to 0 between the up-sweep and the
down-sweep, so the final array is exclusive.

| Stage | Array |
|---|---|
| initial | `[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]` |
| up-sweep d=1 | `[1, 3, 3, 7, 1, 3, 3, 7, 1, 3, 3, 7, 1, 3, 3, 7]` |
| up-sweep d=2 | `[1, 3, 3, 10, 1, 3, 3, 10, 1, 3, 3, 10, 1, 3, 3, 10]` |
| up-sweep d=4 | `[1, 3, 3, 10, 1, 3, 3, 20, 1, 3, 3, 10, 1, 3, 3, 20]` |
| up-sweep d=8 | `[1, 3, 3, 10, 1, 3, 3, 20, 1, 3, 3, 10, 1, 3, 3, 40]` |
| root set to 0 | `[1, 3, 3, 10, 1, 3, 3, 20, 1, 3, 3, 10, 1, 3, 3, 0]` |
| down-sweep d=8 | `[1, 3, 3, 10, 1, 3, 3, 0, 1, 3, 3, 10, 1, 3, 3, 20]` |
| down-sweep d=4 | `[1, 3, 3, 0, 1, 3, 3, 10, 1, 3, 3, 20, 1, 3, 3, 30]` |
| down-sweep d=2 | `[1, 0, 3, 3, 1, 10, 3, 13, 1, 20, 3, 23, 1, 30, 3, 33]` |
| down-sweep d=1 | `[0, 1, 3, 6, 10, 11, 13, 16, 20, 21, 23, 26, 30, 31, 33, 36]` |
| final | `[0, 1, 3, 6, 10, 11, 13, 16, 20, 21, 23, 26, 30, 31, 33, 36]` |

This matches the expected exclusive scan. In the down-sweep, `d=8` has one active block,
`d=4` has two, and so on until `d=1` has eight; the up-sweep visits the same block counts
in the opposite order, so its earliest levels use the most threads and its last level only
one.

## Work and span

For a power-of-two length `N`:

- Up-sweep: `N/2 + N/4 + ... + 1 = N - 1` additions.
- Down-sweep: one exchange and one addition per internal node, again about `N - 1`
  operations.
- Total work `W = Theta(N)` (about `2(N-1)` operations), the same order as the serial
  scan's `n` additions, but with a constant factor near 2.
- Depth/span: `log2 N` up-sweep levels and `log2 N` down-sweep levels, and each level has
  constant depth, so `D = 2 log2 N`.
- With `p` processors the standard work-depth bound is `O(W/p + D) = O(N/p + log N)`.

Two consequences are visible in the measurements:

1. Because the constant is about 2 and the implementation pads `n` up to `N >= n`, the
   parallel version must run at least `2N/n` times faster per unit of work just to tie the
   serial scan. For n = 1e5 that ratio is `2 * 131072 / 100000 = 2.62`, and for n = 1e7 it
   is `2 * 16777216 / 10000000 = 3.36`. On 8 threads the measured `vs_serial` is below 1,
   so the extra work is not recovered.
2. The span bound ignores memory traffic. In this array-layout tree, the active nodes at
   each level are spread across the whole `N`-element tree rather than compacted into
   `N/2^level` contiguous slots. A whole 64-byte cache line is therefore touched for very
   few useful integers, especially at the upper levels. See the derived calculation below.

## Synchronization between stages

There is a barrier after every one of the `2 log2(N)` levels (the implicit barrier at the
end of each `omp for`), plus one after the initial fill and one after `omp single`. For
n = 1e7, `N = 2^24`, so that is 48 level barriers plus 2 = 50 barriers; for n = 1e5,
`N = 2^17`, it is 34 + 2 = 36 barriers.

At **large n** the per-level barrier itself is small compared with the memory traffic of
that level, but the level structure still hurts in two ways:

- Late up-sweep and early down-sweep levels have very few active blocks: level `d` has
  `N/(2d)` blocks, so the last level has 1 block and only one thread does useful work.
  The last `log2(p)` levels cannot use all threads at all.
- Every level starts a new `omp for`, so loop set-up and the barrier are paid 48 times
  even though the total arithmetic is only about 33.5 M updates.

At **small n** the whole function takes 0.19–0.58 ms (n = 1e5), so paying for 36
parallel-loop stages and a fork/join is a significant fraction of the runtime; the serial
scan does 1e5 adds in 0.094 ms in one pass. The barrier cost itself was **not**
instrumented separately, so this explanation is an inference from the measured totals and
the stage count, not a measured breakdown.

## Padding for n = 1025 and in general

- `next_power_of_two(1025) = 2048`.
- Extra elements: `2048 - 1025 = 1023`.
- Overhead: `1023 / 1025 = 0.99805 = 99.805%`, i.e. **about 99.8%** of the original
  length, essentially doubling the array. (The assignment text rounds this to "about
  99.9%"; the exact value is 99.805%.)
- Safe strategy: pad with the identity element `0`. Adding zeros to any prefix leaves the
  prefix sums unchanged, so the first `n` entries of the scanned padded array are exactly
  the requested result. The implementation then truncates the vector back to n, so no
  padded value can appear in the output. This is what `exclusive_scan_parallel` does.
- Worst case: `n = 2^k + 1` gives `N = 2^(k+1)`, so the padded length approaches `2n`.
  n = 1025 is close to that worst case. A blocked two-pass scan (per-thread local scan,
  scan of the block sums, add offsets) needs no padding and no full-tree memory, at the
  cost of a second pass over the data; it is the natural alternative here.
- Measured/derived effect at the benchmark sizes: n = 1e5 pads to 131072 (31.1% extra,
  0.50 MiB tree); n = 1e7 pads to 16,777,216 (67.8% extra, 64 MiB tree). The 64 MiB tree
  is about 2.7x the 25 MiB L3, so the sweep cannot stay in cache.

## Measured results

9 timed runs, medians, `omp_get_wtime()`. `T1` = parallel at 1 thread.

| n | op | variant | p | Tp (ms) | Sp = T1/Tp | Ep = Sp/p | serial (ms) | vs_serial |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1e5 | scan | serial | 1 | 0.094254 | – | – | 0.094254 | 1.000 |
| 1e5 | scan | parallel | 1 | 0.702160 | 1.000 | 1.000 | 0.094254 | 0.134 |
| 1e5 | scan | parallel | 2 | 0.342428 | 2.051 | 1.025 | 0.094254 | 0.275 |
| 1e5 | scan | parallel | 4 | 0.254396 | 2.760 | 0.690 | 0.094254 | 0.371 |
| 1e5 | scan | parallel | 8 | 0.158572 | 4.428 | 0.554 | 0.094254 | 0.594 |
| 1e5 | repeats | serial | 1 | 0.410349 | – | – | 0.410349 | 1.000 |
| 1e5 | repeats | parallel | 1 | 1.841915 | 1.000 | 1.000 | 0.410349 | 0.223 |
| 1e5 | repeats | parallel | 2 | 1.238569 | 1.487 | 0.744 | 0.410349 | 0.331 |
| 1e5 | repeats | parallel | 4 | 0.957811 | 1.923 | 0.481 | 0.410349 | 0.428 |
| 1e5 | repeats | parallel | 8 | 0.830097 | 2.219 | 0.277 | 0.410349 | 0.494 |
| 1e7 | scan | serial | 1 | 39.062097 | – | – | 39.062097 | 1.000 |
| 1e7 | scan | parallel | 1 | 238.780226 | 1.000 | 1.000 | 39.062097 | 0.164 |
| 1e7 | scan | parallel | 2 | 206.969825 | 1.154 | 0.577 | 39.062097 | 0.189 |
| 1e7 | scan | parallel | 4 | 189.146230 | 1.262 | 0.316 | 39.062097 | 0.207 |
| 1e7 | scan | parallel | 8 | 194.364105 | 1.229 | 0.154 | 39.062097 | 0.201 |
| 1e7 | repeats | serial | 1 | 35.670948 | – | – | 35.670948 | 1.000 |
| 1e7 | repeats | parallel | 1 | 314.212601 | 1.000 | 1.000 | 35.670948 | 0.114 |
| 1e7 | repeats | parallel | 2 | 269.393017 | 1.166 | 0.583 | 35.670948 | 0.132 |
| 1e7 | repeats | parallel | 4 | 257.763239 | 1.219 | 0.305 | 35.670948 | 0.138 |
| 1e7 | repeats | parallel | 8 | 235.890687 | 1.332 | 0.167 | 35.670948 | 0.151 |

### Diagnostic decomposition (probe, not part of the graded implementation)

To separate the tree sweep from the per-call `new int[N]` + fill + result copy, a probe
(`scan/probe_scan.cpp`) measured three things in one process per thread count: the
submitted function end to end, the same sweeps on a preallocated tree, and an
allocation+fill+copy with no sweep. 9-run medians:

| n | p | full (ms) | sweep only (ms) | alloc+fill+copy (ms) |
|---:|---:|---:|---:|---:|
| 1e5 | 1 | 0.582 | 0.538 | 0.139 |
| 1e5 | 2 | 0.451 | 0.281 | 0.089 |
| 1e5 | 4 | 0.319 | 0.245 | 0.100 |
| 1e5 | 8 | 0.188 | 0.191 | 0.090 |
| 1e7 | 1 | 246.3 | 174.4 | 79.6 |
| 1e7 | 2 | 205.9 | 171.6 | 62.8 |
| 1e7 | 4 | 195.0 | 154.0 | 53.2 |
| 1e7 | 8 | 188.8 | 156.8 | 53.2 |

The `full` column agrees with the CSV above (within ~3–8% at 1e7, ~10–20% at 1e5, since
the probe uses a different input pattern and ran in a separate process). The three
columns are separate measurements, so they need not add up exactly; the important
observation is that at n = 1e7 roughly **28%** of the end-to-end time at 8 threads
(53 of 189 ms) is allocating, touching, and copying the 64 MiB tree, while the sweep is
about 157 ms.

### Why the sweep itself is slow: cache-line model (derived, not counter-measured)

The tree is a plain array of N ints, and the active nodes of a level are spread through
the whole array. Counting distinct 64-byte cache lines touched (16 ints/line) over both
sweeps for N = 2^24 gives:

- 6,291,454 line touches per sweep, 12,582,908 for up + down.
- At 64 B read + 64 B write-back per touched line (the 64 MiB tree does not fit in L3,
  so there is no level-to-level reuse), that is about **1.61 GB** of line traffic.
- 1.61 GB / 0.157 s = about **10 GB/s** during the preallocated sweep, a plausible DRAM
  bandwidth for this machine.

The serial scan moves about `40 MB read + 40 MB write = 80 MB` in 39.06 ms, which is only
~2 GB/s because a single dependent `sum += input[i]` chain is latency-bound, not
bandwidth-bound. The tree moves roughly **20x more bytes**, and its arithmetic
(`2N ≈ 33.5 M` updates) is also ~3.4x the serial 10 M additions. That is why the parallel
scan loses to serial even at 8 threads, and why the preallocated sweep improves only
1.11x from 1 to 8 threads (174.4 -> 156.8 ms): it is memory-bound, not core-bound.

This calculation assumes 64-byte lines and no reuse between levels; no hardware counters
were read, so the GB/s figure is an estimate derived from the access pattern and the
measured time, not a directly measured bandwidth.

### Does the parallel version beat serial?

In the recorded 9-run benchmark, at **every** size and thread count, **no**: the best
cases are `vs_serial = 0.594`
(scan, 1e5, 8 threads) and `0.494` (repeats, 1e5, 8 threads); at 1e7 the best is `0.207`
(scan) and `0.151` (repeats). So the tree scan scales (up to 4.4x from 1 to 8 threads at
1e5) but starts from a much larger constant, and at 1e7 it barely scales at all
(1.23x from 1 to 8 threads). This is consistent with the work-depth prediction of
`O(N/p + log N)` only in the weak sense that adding threads reduces the time; the
predicted speedup is not reached because the model omits memory traffic, and because the
padding/tree constant is 2.6–3.4x.

`find_repeats_parallel` is slower still because it runs three parallel regions (flags,
scan, scatter), allocates both `flags`/`offsets` and the exact-size result, and the scan
inside it reproduces all of the above. Its measured speedup from 1 to 8 threads is 2.22x
at 1e5 and 1.33x at 1e7, and it never reaches serial.

The 4-thread vs 8-thread points at 1e7 differ in both directions (scan: 189.1 -> 194.4 ms;
repeats: 257.8 -> 235.9 ms). Those differences are only 2–9% on a loaded machine, so they
should not be over-interpreted as either a real gain or a real regression; the defensible
statement is that **there is no clear gain beyond 4 threads at 1e7**, which matches a
saturated memory system.

## Conclusions supported by the measurements

1. Both functions are correct on all tested lengths, including 0 and 1, non-powers of two,
   all-equal/all-distinct/alternating inputs, and fixed-seed random arrays up to 1e7.
   Inputs are never modified.
2. The staged n=16 walkthrough reproduces the expected exclusive scan exactly.
3. In the recorded 9-run benchmark the tree scan scales with threads (Sp = 4.43 at
   n=1e5, p=8) but is slower than the serial scan at every measured point
   (`vs_serial` 0.13–0.59); one interference outlier outside the CSV is noted in the
   limitations.
4. At n=1e7 the preallocated sweep improves only 1.11x from 1 to 8 threads, while the
   serial scan is 4.0x faster than the 8-thread sweep; the measured `vs_serial` is 0.20.
5. About 28% of the n=1e7 end-to-end time at 8 threads is per-call tree allocation, fill,
   and result copy (53 of 189 ms), measured by the probe.
6. The access-pattern calculation predicts ~1.61 GB of cache-line traffic for the two
   sweeps at n=1e7, about 20x the serial scan's 80 MB, consistent with the measured
   ~10 GB/s sweep rate.
7. Padding is exactly 1023 extra elements (99.8%) at n=1025 and 67.8% at n=1e7; it is
   handled safely with zeros and truncation, but it is part of why the constant factor is
   large.

## Assumptions and limitations (not directly measured)

- The split between barrier/`omp for` cost and memory traffic is not instrumented.
  Statements about barriers dominating at small n, or about idle threads in the last
  `log2 p` levels, are inferences from the stage structure and measured totals.
- The cache-line traffic estimate assumes 64 B lines, a write-back read-modify-write per
  line, and no reuse between levels. It is consistent with the measured time, but no
  performance counters were used.
- The 1e5 probe's end-to-end number differs from the CSV by up to 20% because the two
  ran in separate processes with different input patterns; only the within-probe
  comparison is used.
- The machine was shared and under memory pressure; absolute times and the 4 vs 8 thread
  ordering at 1e7 would differ on an idle machine. The qualitative result (slower than
  serial, memory-bound at 1e7, better scaling when the tree fits in L3 at 1e5) was
  reproduced in repeated runs. One separate 1-run check of the supplied benchmark during
  a higher load window measured the n=1e5 serial scan at 0.294 ms instead of 0.094 ms,
  clearly an interference outlier on that core; the 9-run medians in `scan_results.csv`
  are the recorded values. Small-n `vs_serial` numbers therefore carry more uncertainty
  than the n=1e7 numbers, where the gap is large (0.20) and stable.
- Only 1/2/4/8 threads were measured; 10 physical cores were not fully used and
  hyper-thread oversubscription was not tested.
