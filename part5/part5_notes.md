# Part 5 notes: MPI collectives and communication-cost reasoning

This part is analytical: there is no MPI source code and nothing is measured. Every
number below is recomputed by `part5/verify_part5.py`; its captured output is
`part5/part5_output.txt` (57 checks, all OK). The two diagrams are:
`part5/part5_dataflow.png` and `part5/part5_topologies.png`. They are drawn by
`part5/draw_part5_diagrams.py` and are model illustrations, not measurements.

## Assumptions used throughout

- Units: time in microseconds (us) unless stated; 1 ms = 1000 us.
- Message model: `T(m) = alpha + beta*m` with `alpha = 10 us` and
  `beta = 1 ns/byte` (so `beta = 0.001 us/byte`).
- Links are contention-free and full-duplex, with no pipelining or segmentation.
  The linear and binomial broadcast derivations use the single-port model: a process
  sends or receives at most one message at a time. Store-and-forward means a process
  forwards only after receiving the whole message.
- Topology step counts (task D) use the requested "one step = one round of
  simultaneous sends along direct links" convention. They are diameter-based; a node
  is allowed to forward along its outgoing links in the same round. A strict one-port
  restriction would add serialization at nodes of degree greater than 2. This is
  stated again where it matters; it does not change the expected diameter counts.
- No communication/computation overlap is assumed in the task-E cost comparison.
- Sizes used below:
  - circle record: 7 float32 fields `(x, y, r, red, green, blue, alpha)` = 28 bytes;
  - scene metadata: four int32 fields `(width, height, tile_size, Nc)` = 16 bytes;
  - RGBA float32 pixel = 16 bytes; 1024x1024 image = 16 MiB;
  - 16x16 RGBA tile = 4096 bytes; 32x32 RGBA tile = 16,384 bytes.

---

## A. Data-flow design

### A.1 Design

The root rank loads the scene. It then broadcasts the scene metadata and the full
circle array. In the simple design there are two broadcast calls (or one packed
broadcast):

1. metadata: `width`, `height`, `tile_size`, `Nc` (16 bytes);
2. circles: `Nc` records of 28 bytes, containing position, radius and RGBA.

Every rank therefore has enough information to run the same deterministic tile
assignment and the same Part 4 preprocessing (padded circle boxes and ordered
per-tile candidate lists). Tile ownership is assigned without a separate tile-data
broadcast:

- **Recommended default:** rank `r` owns tiles `r, r+p, r+2p, ...` (round-robin).
  Every rank can compute this from `r`, `p` and the tile count, so a tile-ID scatter
  is optional.
- **Equivalent scatter form:** root owns the global tile-ID list and partitions it,
  then sends the ID list with `MPI_Scatter` (equal counts) or `MPI_Scatterv`
  (different counts). This is useful for weighted or irregular assignments, or if
  the assignment policy is changed without changing every rank's code.

For each owned tile, the rank runs the Part 4 renderer locally: it uses the tile's
ordered candidate list, calls the exact `covers()`/`blend()` for each candidate, and
writes the tile's pixels. Tiles are disjoint, so no atomics or locks are needed.

The finished tile pixels are returned in tile order with `MPI_Gather` or
`MPI_Gatherv` onto root. Root copies each tile into its `(tx, ty)` rectangle in the
final image. Root can allocate the final 16 MiB image before the gather so the copy
has a destination ready.

The flow is shown in `part5/part5_dataflow.png`:

```
root: load scene
  -> MPI_Bcast metadata + circles
  -> ranks receive scene, derive or scatter tile IDs
  -> each rank builds per-tile candidate lists and renders its tiles
  -> MPI_Gatherv tile buffers in tile order
  -> root reassembles (tx,ty) rectangles and writes the image
```

### A.2 Why `MPI_Scatterv` / `MPI_Gatherv` are needed

`MPI_Scatter` and `MPI_Gather` require the same count on every rank. That is not
true here whenever:

- the number of tiles is not divisible by `p`;
- the image size is not divisible by the tile size, so edge tiles have fewer
  pixels (and a different payload size);
- tile assignment is uneven or weighted, so ranks own different numbers of tiles.

The variable-count collectives take two integer arrays:

- `counts[r]`: number of elements sent/received by rank `r`;
- `displs[r]`: offset in the global send/recv buffer where rank `r`'s block starts.

**Concrete example from the script:** a 1050x1050 image with 16-pixel tiles.

- tiles per side = `ceil(1050/16) = 66`;
- total tiles = `66 * 66 = 4356`;
- block split over `p = 8`: `4356 // 8 = 544` with remainder 4;
- `counts[] = [545, 545, 545, 545, 544, 544, 544, 544]`;
- `displs[] = [0, 545, 1090, 1635, 2180, 2724, 3268, 3812]`;
- the last rank ends at `3812 + 544 = 4356 = total tiles`.

The last column and last row are 10 pixels wide/tall (`1050 = 65*16 + 10`), so edge
tiles have smaller payloads than a full 16x16 tile. If `counts`/`displs` are in bytes,
those edge sizes must be used; if they are in whole-tile units, the receiver must
know each tile's actual size (or every tile must be padded to the full 4096 bytes).
The former is more compact; the latter makes the count arithmetic simpler.

For a rectangular block assignment, `MPI_Scatterv` can send contiguous tile-ID
blocks. For the round-robin assignment `r, r+p, ...`, each rank has a strided list;
that is fine for a tile-ID list, but the rendered tile buffers are simply packed in
the rank's own receive order and root uses the global tile IDs when reassembling.

### A.3 Non-contiguous tiles: gather in tile order, reassemble on root

Tiles are non-contiguous in the final image: tile `(tx, ty)` is a small rectangle in
a row-major image. Two valid implementations are:

1. **Gather tile buffers in tile order and copy on root (chosen).** Each rank packs
   its rendered tiles in ascending global tile order into one contiguous send buffer.
   `Gatherv` concatenates the rank blocks; root walks the global tile-ID list and
   copies each tile's pixels into its `(tx, ty)` rectangle.
2. **Use `MPI_Type_create_subarray` or an explicit pack/unpack.** A subarray
   datatype can describe a rectangular block in the final image, and a derived type
   or `MPI_Pack` can avoid the root-side copy.

I choose option 1 because it is simpler and robust to edge tiles and irregular
ownership. It also keeps a single explicit tile order that is easy to validate against
the sequential reference. The cost is one extra copy on root; the image is 16 MiB at
1024x1024, which is not large relative to the model's communication times. A derived
subarray is worthwhile in a production code if the root copy becomes measurable, but
it does not change the algorithm or the order of blending.

### A.4 Full-circle broadcast versus overlap-only sends

**Recommendation: broadcast the full circle array.** It is one small message and it
makes every rank independent and deterministic:

- `Nc = 400`: `400 * 28 = 11,200` bytes; `T = 10 + 11,200*0.001 = 21.200 us`
  per message.
- `Nc = 2000`: `2000 * 28 = 56,000` bytes; `T = 10 + 56,000*0.001 = 66.000 us`
  per message.

The alternative is for root to determine which circles overlap which rank's tiles and
send only those. That reduces bytes for sparse scenes, but requires root-side tile
preprocessing, a different message per rank, and either point-to-point messages or a
variable-count scatter. It also makes the per-rank data depend on the assignment and
adds a failure/validation surface. With 28-byte records and at most tens of kilobytes in
these scenes, the full broadcast is the better trade-off. Use overlap-only sending only
if `Nc` is very large or the circle data is much larger than the tile payloads.

### A.5 Data-size summary

| item | size |
|---|---:|
| metadata (4 int32) | 16 B |
| circle record (7 float32) | 28 B |
| circles, `Nc=400` | 11,200 B |
| circles, `Nc=2000` | 56,000 B |
| RGBA float32 pixel | 16 B |
| 1024x1024 image | 16,777,216 B = 16 MiB |
| 16x16 tile | 4,096 B |
| tiles at 1024x1024, 16-pixel tiles | 4,096 |
| tiles per rank at `p=8` (1024²) | 512 |
| total gathered tile bytes at 1024² | 16,777,216 B = 16 MiB |
| 32x32 tile (task E assumption) | 16,384 B |

The full-circle broadcast is 16 B plus `Nc*28` B. For `Nc=2000` that is 56,016 B
in a single metadata+array broadcast (or 56,000 B if the metadata is sent separately).

---

## B. Broadcast cost derivations under `T(m) = alpha + beta*m`

### B.1 Linear broadcast

Root sends the message sequentially to each of the `p-1` other processes. In the
single-port model these sends do not overlap, so the times add:

```
T_linear = sum over p-1 messages of (alpha + beta*m)
         = (p - 1) * (alpha + beta*m).
```

For `p = 8`, this is `7 * T(m)`.

### B.2 Binomial tree broadcast

Let `h = ceil(log2 p)`. Initially only root holds the message. In each round every
process that already holds the message sends exactly one copy to a process that does
not yet hold it. The number of holders doubles each round:

```
round 0: 1 holder
round 1: 2 holders
round 2: 4 holders
...
round r: 2^r holders (until p)
```

After `h` rounds all `p` processes hold the data. The critical path is one message per
round, so in the single-port model:

```
T_binomial = h * (alpha + beta*m)
           = ceil(log2 p) * (alpha + beta*m).
```

For `p = 8`, `h = ceil(log2 8) = 3`, so `T_binomial = 3 * T(m)`.

One valid `p=8` schedule (the one printed by the script) is:

| round | senders -> receivers | holders after the round |
|---:|---|---:|
| 0 | (root 0 already has it) | 1 |
| 1 | `0 -> 4` | 2 |
| 2 | `0 -> 2`, `4 -> 6` | 4 |
| 3 | `0 -> 1`, `2 -> 3`, `4 -> 5`, `6 -> 7` | 8 |

Every sender in a round was a holder at the start of that round, and every receiver
was not a holder before it. Any equivalent pairing is valid; the count is the same.
For a non-power-of-two `p`, some processes are idle in the last round, but the number
of rounds is still `ceil(log2 p)`.

### B.3 Large-message note

The binomial tree is a latency-oriented construction: it reduces the number of
sequential rounds from `p-1` to `ceil(log2 p)`, but each of the `p-1` tree edges still
carries a full `m`-byte message. It is not bandwidth-optimal for very large `m`.
Scatter+allgather, recursive halving/doubling, or pipelining/segmentation can overlap
message pieces and use more links. That is a note only; the main calculation below
uses the linear-versus-binomial model as requested.

---

## C. Numerical calculation: `p = 8`, `m = 2^20` B, `alpha = 10 us`, `beta = 1 ns/B`

### C.1 One message

```
m       = 2^20 = 1,048,576 bytes
beta*m  = 1 ns/byte * 1,048,576 bytes
        = 1,048,576 ns
        = 1,048,576 / 1000 us
        = 1048.576 us
T(m)    = alpha + beta*m
        = 10 us + 1048.576 us
        = 1058.576 us.
```

### C.2 Linear and binomial

```
linear   = (p-1) * T(m) = 7 * 1058.576 us
         = 7410.032 us
         = 7.410032 ms.

binomial = ceil(log2 p) * T(m) = 3 * 1058.576 us
         = 3175.728 us
         = 3.175728 ms.
```

### C.3 Ratio and interpretation

```
ratio = linear / binomial = 7410.032 / 3175.728 = 7/3 = 2.3333...
```

The binomial tree saves four message times:

```
linear - binomial = 7410.032 - 3175.728 = 4234.304 us.
```

Of that saving, the latency part is only `4 * alpha = 4 * 10 = 40 us`, while the
bandwidth part is `4 * beta*m = 4 * 1048.576 = 4194.304 us`. Equivalently:

```
alpha / T(m)    = 10 / 1058.576       = 0.9447%
beta*m / T(m)   = 1048.576 / 1058.576 = 99.0553%.
```

The effective one-message bandwidth is

```
m / T(m) = 1,048,576 B / 1058.576 us = 990.553 B/us = 990.553 MB/s.
```

So this case is strongly **bandwidth-dominated**. Halving the number of rounds is
valuable, but the reason is that fewer full-message transmissions sit on the critical
path; the alpha-only saving is under 1% of the total saving. A low-latency network
would not change the order of magnitude here.

---

## D. Broadcast steps on ring, 2D mesh and hypercube for `p = 16`

### D.1 Model and routing assumptions

- `p = 16`, ranks `0..15`, root = rank 0.
- Store-and-forward: a node receives the whole message before forwarding it.
- One step = one round of simultaneous sends along direct links.
- Bidirectional links; every link has the same speed.
- No contention and no pipelining. Step counts are latency/diameter counts; the time
  of one step is modeled as `alpha + beta*m` separately (see the table below).
- For the diameter counts, a node may forward to all of its outgoing neighbors in the
  same round. Under a strict one-port constraint, a high-degree node would need extra
  rounds to serialize its outgoing messages. This caveat matters for the mesh; the
  binomial schedule on a hypercube already sends one message per holder per round.

### D.2 Ring, 16 nodes

**Unidirectional ring:** the only path is

```
0 -> 1 -> 2 -> ... -> 15
```

which uses 15 links. Node `i` is reached in step `i`, so the last node needs 15 steps.

**Bidirectional ring:** root sends both ways in round 1:

```
round 1: 0 -> 1 and 0 -> 15
round 2: 1 -> 2 and 15 -> 14
...
round r: r-1 -> r and 17-r -> 16-r
```

After `r` rounds the reached nodes are `1..r` and `16-r..15`. For `r = 8` both sides
have met: `1..8` and `8..15` cover all 16 nodes (node 8 is reached at step 8 from both
directions). The maximum distance is 8, i.e.

```
ceil((p-1)/2) = ceil(15/2) = 8 steps.
```

Model time at `m = 2^20`: `8 * 1058.576 = 8468.608 us`.

### D.3 2D mesh 4x4, no wraparound

Use coordinates `(x,y)` with `x,y` in `0..3`; root at a corner `(0,0)`. A message
moves one hop horizontally or vertically per step, so the step at which `(x,y)` is
reached is its Manhattan distance from root:

```
s(x,y) = |x - 0| + |y - 0| = x + y.
```

The farthest node is the opposite corner `(3,3)`:

```
s_max = 3 + 3 = 6  =>  diameter = 6 steps.
```

The wavefront is:

| step | nodes reached (x+y = step) |
|---:|---|
| 0 | (0,0) |
| 1 | (1,0), (0,1) |
| 2 | (2,0), (1,1), (0,2) |
| 3 | (3,0), (2,1), (1,2), (0,3) |
| 4 | (3,1), (2,2), (1,3) |
| 5 | (3,2), (2,3) |
| 6 | (3,3) |

Model time: `6 * 1058.576 = 6351.456 us`.

**Root at the centre:** put root at `(1,1)`. The farthest corners are at
`|dx| + |dy| = 2 + 2 = 4`, so the diameter from that root is 4 steps. Model time:
`4 * 1058.576 = 4234.304 us`.

**Torus (wraparound):** in a 4x4 torus each dimension has diameter `floor(4/2) = 2`,
so the worst-case distance is `2 + 2 = 4` steps, including from a corner. Model time:
`4 * 1058.576 = 4234.304 us`.

### D.4 Hypercube Q4

`16 = 2^4`, so the hypercube has dimension 4. Label nodes by 4-bit ranks. Recursive
doubling fixes one bit per round:

| round | bit fixed | informed nodes | why |
|---:|---|---:|---|
| 1 | b0 | 2 | each holder sends to the node differing in b0 |
| 2 | b1 | 4 | all nodes with b0 fixed are now holders |
| 3 | b2 | 8 | each holder covers one value of b2 |
| 4 | b3 | 16 | all 16 nodes covered |

The step at which a node is reached is its Hamming distance from root `0000`, i.e. the
number of 1 bits in its rank. Thus the diameter is `log2(16) = 4` steps. Model time:
`4 * 1058.576 = 4234.304 us`.

The topology diagram `part5/part5_topologies.png` marks, for each node, the round `sN`
in which it is reached. For the hypercube, the edge color identifies which bit is
fixed in that round.

### D.5 Step-count summary

| topology / root | steps | derivation | model time at `m=2^20` |
|---|---:|---|---:|
| ring, unidirectional | 15 | `0->1->...->15` | 15 * 1058.576 = 15,878.640 us |
| ring, bidirectional | 8 | diameter `16/2 = 8`; root sends both ways | 8 * 1058.576 = 8,468.608 us |
| 2D mesh 4x4, corner root | 6 | Manhattan `3+3` | 6 * 1058.576 = 6,351.456 us |
| 2D mesh 4x4, centre root | 4 | max Manhattan from `(1,1)` is `2+2` | 4 * 1058.576 = 4,234.304 us |
| 2D torus 4x4 | 4 | `2+2` with wraparound | 4 * 1058.576 = 4,234.304 us |
| hypercube Q4 | 4 | `log2(16) = 4` | 4 * 1058.576 = 4,234.304 us |

These times are **model-based**, not measured. The times assume every step has the
same full-message cost `T(m) = alpha + beta*m`.

### D.6 Why step count alone may not predict real time

The step counts are a latency model, not a performance measurement:

- **Message size:** a step costs `alpha + beta*m`. For large `m`, `beta*m` dominates
  and the number of bytes moved matters as much as the number of rounds.
- **Link contention and bisection bandwidth:** simultaneous messages share links. A
  ring has a small bisection; a mesh and a hypercube have larger bisections. A
  wavefront can look short in steps but serialize on a hot link.
- **Non-uniform link speeds:** links may differ, so equal-cost steps are an
  idealization.
- **Logical-to-physical mapping:** MPI ranks are not guaranteed to map one-to-one onto
  the physical topology. Placement and oversubscription can add hops or sharing.
- **Real cluster topology:** production clusters commonly use switched fat-trees, not
  rings or meshes. A logical ring pattern may map onto a tree with different
  contention, and a logical hypercube may not match the cabling.
- **Pipelining/segmentation:** real networks cut messages into packets; latency and
  bandwidth overlap. Store-and-forward whole-message forwarding is a simplification.
- **Forwarding latency and software overhead:** MPI buffering, protocol processing,
  copies, and rendezvous/eager thresholds add per-message costs not in `alpha+beta*m`.
- **Single-port assumption:** a high-degree node cannot send to all neighbors at once
  under strict one-port; extra rounds may be needed and the diameter count becomes a
  lower bound.

Nothing in this section is measured.

---

## E. Load balance with tile times `[2, 2, 3, 3, 3, 4, 5, 18]` ms on 8 processes

### E.1 Current assignment: one tile per process

```
tile times = [2, 2, 3, 3, 3, 4, 5, 18] ms
total work = 2+2+3+3+3+4+5+18 = 40 ms
p          = 8
```

The average-load lower bound is

```
40 / 8 = 5 ms.
```

But tiles are indivisible and one tile is 18 ms, so the true lower bound with this
granularity is

```
max(average, largest tile) = max(5, 18) = 18 ms.
```

With one tile per process, the makespan is the maximum assigned time:

```
makespan = max(t_i) = 18 ms.
```

Therefore, ignoring communication:

```
speedup    = total work / makespan = 40 / 18 = 2.2222...
efficiency = total work / (p * makespan) = 40 / (8 * 18)
           = 40 / 144 = 0.277777... = 27.7778%.
```

Equivalently, `efficiency = speedup / p = (40/18)/8 = 40/144`.

Idle time:

```
total idle = p * makespan - total work
           = 8 * 18 - 40
           = 144 - 40
           = 104 process-ms.
```

Per-rank idle times `makespan - t_i` are

```
[16, 16, 15, 15, 15, 14, 13, 0]  (sum = 104 process-ms).
```

### E.2 Recommended change: over-decompose and assign dynamically with LPT

**One change:** split the 18 ms tile into smaller pieces and use over-decomposition
with dynamic assignment. A concrete, simple version is to split it into four equal
4.5 ms pieces and use LPT (longest processing time first) over all pieces. LPT is a
static greedy simulation of a work queue: sort jobs largest first and assign each to
the currently least-loaded process (ties to the lowest rank).

Assumption: the 18 ms tile divides evenly into four 4.5 ms sub-tiles. In a real
renderer, sub-tile cost may not divide exactly, so this is an idealization; the
qualitative result (the 18 ms straggler is gone) is what matters.

After splitting, the jobs are

```
[2, 2, 3, 3, 3, 4, 5, 4.5, 4.5, 4.5, 4.5]  (11 jobs).
```

In LPT order:

```
[5, 4.5, 4.5, 4.5, 4.5, 4, 3, 3, 3, 2, 2].
```

The simulation assigns them as follows (loads are the eight process loads after each
assignment):

| step | job (ms) | rank | loads after assignment |
|---:|---:|---:|---|
| 0 | 5 | 0 | `[5, 0, 0, 0, 0, 0, 0, 0]` |
| 1 | 4.5 | 1 | `[5, 4.5, 0, 0, 0, 0, 0, 0]` |
| 2 | 4.5 | 2 | `[5, 4.5, 4.5, 0, 0, 0, 0, 0]` |
| 3 | 4.5 | 3 | `[5, 4.5, 4.5, 4.5, 0, 0, 0, 0]` |
| 4 | 4.5 | 4 | `[5, 4.5, 4.5, 4.5, 4.5, 0, 0, 0]` |
| 5 | 4 | 5 | `[5, 4.5, 4.5, 4.5, 4.5, 4, 0, 0]` |
| 6 | 3 | 6 | `[5, 4.5, 4.5, 4.5, 4.5, 4, 3, 0]` |
| 7 | 3 | 7 | `[5, 4.5, 4.5, 4.5, 4.5, 4, 3, 3]` |
| 8 | 3 | 6 | `[5, 4.5, 4.5, 4.5, 4.5, 4, 6, 3]` |
| 9 | 2 | 7 | `[5, 4.5, 4.5, 4.5, 4.5, 4, 6, 5]` |
| 10 | 2 | 5 | `[5, 4.5, 4.5, 4.5, 4.5, 6, 6, 5]` |

Final loads:

```
[5.0, 4.5, 4.5, 4.5, 4.5, 6.0, 6.0, 5.0]
```

The makespan is 6 ms, so

```
compute saved = 18 - 6 = 12 ms
speedup       = 40 / 6 = 6.6667...
efficiency    = 40 / (8 * 6) = 40 / 48 = 0.833333... = 83.3333%.
```

A refined argument shows that 6 ms is in fact optimal for this split. A makespan below
6 ms is impossible: each 4.5 ms piece would have to sit alone (adding any other job,
which is at least 2 ms, would exceed 6 ms), using four ranks. The 5 ms job also sits
alone (5+2 = 7), leaving `4+3+3+3+2+2 = 17 ms` for three ranks. The 4 ms job then sits
alone too (4+2 = 6), so `3+3+3+2+2 = 13 ms` would need two ranks, but two ranks below
6 ms can hold less than 12 ms. Hence no makespan below 6 ms is feasible, and LPT's
6 ms is optimal for this split.

### E.3 Communication cost of the change

Assumptions for this comparison:

- a tile payload of 32x32 RGBA float32 = `32*32*16 = 16,384 bytes`;
- a request message of 16 bytes;
- `alpha = 10 us`, `beta = 1 ns/byte`;
- a master-worker request/reply per tile, with no overlap between communication and
  computation, and no contention;
- the original 40 ms work figure is computation only.

Per-tile messages:

```
request: T(16)     = 10 + 0.016 = 10.016 us
reply:   T(16,384) = 10 + 16.384 = 26.384 us
request + reply    = 36.400 us
```

There are 11 tiles after the split (8 original, with the 18 ms tile replaced by four
4.5 ms pieces):

```
queue cost = 11 * 36.400 = 400.400 us = 0.4004 ms.
```

As a share of the new 6 ms makespan:

```
0.4004 / 6 = 0.066733... = 6.6733%.
```

The net gain is still large:

```
12 ms - 0.4004 ms = 11.5996 ms.
```

A batched gather instead of a request/reply per tile moves `11 * 16,384 = 180,224 B`;
a rough tree/batch estimate is `3*alpha + beta*bytes = 30 + 180.224 = 210.224 us`
(that is an approximation, not a full gather model). The break-even number of
per-tile queue messages for the 12 ms compute saving is

```
12,000 us / 36.400 us = 329.67 tiles.
```

At 1,024 tiles (a 32x32 split of 1024x1024), the per-tile request/reply alone would
cost `1024 * 36.400 = 37,273.6 us = 37.3 ms`, which is larger than the compute saved.
So the recommendation is **over-decompose modestly, then batch the gather** (a single
`Gatherv` or a blocked work queue) rather than using one request/reply per tiny tile.

The extra communication also includes more `counts`/`displs` entries, any border or
metadata duplication, and root contention if many workers request work at once. The
compute saving (12 ms) is much larger than the modeled 0.4004 ms queue cost, so the
change is worthwhile for this skewed distribution. For a balanced distribution, the
same over-decomposition has much less to gain and its message overhead could dominate.

---

## Summary of final numbers

- Circle record: 28 B; `Nc=400` -> 11,200 B; `Nc=2000` -> 56,000 B.
- 1024x1024 RGBA float32 image: 16 MiB; 16x16 tile: 4,096 B; 32x32 tile: 16,384 B.
- Scatterv example (1050x1050, 16-pixel tiles): 4,356 tiles;
  `counts=[545,545,545,545,544,544,544,544]`,
  `displs=[0,545,1090,1635,2180,2724,3268,3812]`.
- `m = 2^20 = 1,048,576 B`; `beta*m = 1048.576 us`;
  `T(m) = 1058.576 us`.
- Linear broadcast (p=8): 7,410.032 us = 7.410032 ms.
- Binomial broadcast (p=8): 3,175.728 us = 3.175728 ms.
- Linear/binomial ratio: 7/3 = 2.3333; bandwidth share of one message: 99.0553%.
- Step counts (p=16): ring unidirectional 15; ring bidirectional 8; mesh corner 6;
  mesh centre 4; torus 4; Q4 hypercube 4.
- Task E original: total 40 ms; average bound 5 ms; indivisible bound 18 ms;
  makespan 18 ms; speedup 2.2222x; efficiency 27.7778%; idle 104 process-ms.
- Task E LPT: makespan 6 ms; speedup 6.6667x; efficiency 83.3333%; compute saved 12 ms.
- Task E communication: 36.400 us per request/reply; 11 tiles -> 400.400 us
  (0.4004 ms); net gain 11.5996 ms; batch estimate 210.224 us; break-even 329.67
  per-tile messages.

All of the above are verified by `part5/verify_part5.py`; rerun it to reproduce the
values and the assertion checks.
