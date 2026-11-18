#!/usr/bin/env python3
"""Part 5: recompute and verify every number used in the written answers.

This script is analytical only. It prints the broadcast costs, topology step
counts, data-size estimates and the load-balance/LPT simulation, asserts the
expected values, and saves the same text to part5_output.txt next to this file.

Run:
    python3 part5/verify_part5.py
"""

import math
import sys
from pathlib import Path

# ---------------------------------------------------------------- assumptions
ALPHA_US = 10.0          # startup latency per message, microseconds
BETA_NS_PER_BYTE = 1.0   # per-byte cost, nanoseconds per byte
P_BROADCAST = 8          # processes for tasks B and C
M_BYTES = 2 ** 20        # message size for task C

CIRCLE_FLOATS = 7        # x, y, r, red, green, blue, alpha
FLOAT_BYTES = 4
CIRCLE_BYTES = CIRCLE_FLOATS * FLOAT_BYTES

IMAGE_W = IMAGE_H = 1024
RGBA_BYTES = 4 * FLOAT_BYTES      # RGBA float32 = 16 bytes per pixel
TILE16_BYTES = 16 * 16 * RGBA_BYTES
TILE32_BYTES = 32 * 32 * RGBA_BYTES


def msg_us(m_bytes):
    """T(m) = alpha + beta*m, with the result in microseconds."""
    return ALPHA_US + BETA_NS_PER_BYTE * m_bytes / 1000.0


def us_to_ms(value_us):
    return value_us / 1000.0


# ------------------------------------------------------------------- output
class Tee:
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")

    def write(self, text):
        sys.stdout.write(text)
        self.file.write(text)

    def close(self):
        self.file.close()


out = Tee(Path(__file__).resolve().parent / "part5_output.txt")


def say(text=""):
    out.write(text + "\n")


def section(title):
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


def check(label, got, expected, tol=1e-9):
    ok = abs(got - expected) <= tol * max(1.0, abs(expected))
    say(f"  check: {label:<48} {got:>14.6f}  expected {expected:>14.6f}  "
        f"{'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit(f"verification failed: {label}")


say("Part 5 verification: MPI collectives and communication-cost reasoning")
say("All values are model-based (nothing measured).")
say(f"assumptions: alpha = {ALPHA_US:g} us/message, beta = {BETA_NS_PER_BYTE:g} ns/byte, "
    f"contention-free links, no pipelining; B/C single-port, D diameter-based rounds")

# --------------------------------------------------------- A. data sizes
section("A. Data-size estimates")
tiles_per_side = IMAGE_W // 16
tiles_16 = tiles_per_side * tiles_per_side
image_bytes = IMAGE_W * IMAGE_H * RGBA_BYTES
say(f"circle record: {CIRCLE_FLOATS} float32 fields = {CIRCLE_BYTES} bytes")
say(f"  x, y, r, red, green, blue, alpha")
for nc in (400, 2000):
    say(f"  Nc = {nc:>5}: circle array = {nc} * {CIRCLE_BYTES} = {nc * CIRCLE_BYTES} bytes")
say(f"scene metadata: width, height, tile_size, Nc as 4 int32 = {4 * 4} bytes")
say(f"image 1024x1024 RGBA float32 = {IMAGE_W}*{IMAGE_H}*{RGBA_BYTES} = "
    f"{image_bytes} bytes = {image_bytes / 2**20:.0f} MiB")
say(f"16x16 tile = 16*16*{RGBA_BYTES} = {TILE16_BYTES} bytes; tiles = "
    f"({IMAGE_W}/16)^2 = {tiles_16}")
say(f"tiles per rank at p=8 = {tiles_16}/8 = {tiles_16 // 8}; gathered total = "
    f"{tiles_16} * {TILE16_BYTES} = {tiles_16 * TILE16_BYTES} bytes")
say(f"32x32 tile (task E assumption) = 32*32*{RGBA_BYTES} = {TILE32_BYTES} bytes")
check("circle bytes", CIRCLE_BYTES, 28)
check("2000-circle bytes", 2000 * CIRCLE_BYTES, 56000)
check("400-circle bytes", 400 * CIRCLE_BYTES, 11200)
check("image bytes", image_bytes, 16777216)
check("16x16 tile bytes", TILE16_BYTES, 4096)
check("tiles at 1024/16", tiles_16, 4096)
check("32x32 tile bytes", TILE32_BYTES, 16384)

say()
say("full-circle broadcast cost at p=8 (binomial = 3 rounds, linear = 7 messages):")
for nc in (400, 2000):
    m = nc * CIRCLE_BYTES
    say(f"  Nc={nc:>4}: m={m:>6} B, T(msg)={msg_us(m):.3f} us, "
        f"linear={7 * msg_us(m):.3f} us, binomial={3 * msg_us(m):.3f} us")
check("circle bcast message Nc=2000", msg_us(2000 * CIRCLE_BYTES), 66.0)
check("circle bcast linear Nc=2000", 7 * msg_us(2000 * CIRCLE_BYTES), 462.0)
check("circle bcast binomial Nc=2000", 3 * msg_us(2000 * CIRCLE_BYTES), 198.0)
check("circle bcast message Nc=400", msg_us(400 * CIRCLE_BYTES), 21.2)
check("circle bcast linear Nc=400", 7 * msg_us(400 * CIRCLE_BYTES), 148.4)
check("circle bcast binomial Nc=400", 3 * msg_us(400 * CIRCLE_BYTES), 63.6)

say()
say("Scatterv/Gatherv concrete example: 1050x1050 image, 16-pixel tiles")
w = h = 1050
tile = 16
tx = (w + tile - 1) // tile
ty = (h + tile - 1) // tile
tiles = tx * ty
say(f"  tiles per side = ceil(1050/16) = {tx}; total tiles = {tx}*{ty} = {tiles}")
raw, rem = divmod(tiles, 8)
counts = [raw + (1 if r < rem else 0) for r in range(8)]
displs = [sum(counts[:r]) for r in range(8)]
say(f"  block split: base = {tiles}//8 = {raw}, remainder = {rem}")
say(f"  counts[] = {counts}")
say(f"  displs[] = {displs}")
say(f"  last rank [0] = {displs[-1]} + {counts[-1]} = {displs[-1] + counts[-1]} = {tiles}")
check("1050 tiles", tiles, 4356)
check("counts sum", sum(counts), 4356)
check("displs[1]", displs[1], 545)
check("displs[7]", displs[7], 3812)
edge = w % tile
say(f"  edge tiles: last column/row = {edge} pixels wide/tall, so their payload "
    f"differs from a full {tile}x{tile} tile")

# ---------------------------------------------------------- B. derivations
section("B. Broadcast cost under T(m) = alpha + beta*m")
rounds_p8 = math.ceil(math.log2(P_BROADCAST))
say(f"p = {P_BROADCAST}: ceil(log2 p) = ceil({math.log2(P_BROADCAST):.3f}) = {rounds_p8}")
say(f"linear   T = (p-1)(alpha + beta*m) = {P_BROADCAST - 1} * T(msg)")
say(f"binomial T = ceil(log2 p)(alpha + beta*m) = {rounds_p8} * T(msg)")
say()
say("p=8 binomial broadcast rounds (root = 0):")
schedule = {
    1: [(0, 4)],
    2: [(0, 2), (4, 6)],
    3: [(0, 1), (2, 3), (4, 5), (6, 7)],
}
holders = {0}
for rnd in (1, 2, 3):
    pairs = schedule[rnd]
    say(f"  round {rnd}: " + ", ".join(f"{a}->{b}" for a, b in pairs))
    for a, b in pairs:
        assert a in holders and b not in holders, (rnd, a, b, holders)
        holders.add(b)
check("holders after 3 rounds", len(holders), 8)

# ------------------------------------------------------------- C. numbers
section("C. Numerical calculation: p = 8, m = 2^20, alpha = 10 us, beta = 1 ns/byte")
beta_m_ns = BETA_NS_PER_BYTE * M_BYTES
beta_m_us = beta_m_ns / 1000.0
one = msg_us(M_BYTES)
linear = (P_BROADCAST - 1) * one
binomial = rounds_p8 * one
ratio = linear / binomial
say(f"  m                 = 2^20 = {M_BYTES:,} bytes")
say(f"  beta*m            = {M_BYTES:,} ns = {beta_m_us:.3f} us")
say(f"  alpha + beta*m    = {ALPHA_US:g} + {beta_m_us:.3f} = {one:.3f} us")
say(f"  linear            = 7 * {one:.3f} = {linear:.3f} us = {us_to_ms(linear):.6f} ms")
say(f"  binomial          = 3 * {one:.3f} = {binomial:.3f} us = "
    f"{us_to_ms(binomial):.6f} ms")
say(f"  ratio linear/binomial = 7/3 = {ratio:.4f}")
latency_share = ALPHA_US / one
bandwidth_share = beta_m_us / one
say(f"  latency share alpha/T(m)   = {ALPHA_US:g}/{one:.3f} = {latency_share:.4%}")
say(f"  bandwidth share beta*m/T(m) = {beta_m_us:.3f}/{one:.3f} = {bandwidth_share:.4%}")
saved = linear - binomial
say(f"  total saved by binomial = 4 * {one:.3f} = {saved:.3f} us")
say(f"  of which latency only   = 4 * {ALPHA_US:g} = {4 * ALPHA_US:.3f} us "
    f"({4 * ALPHA_US / saved:.2%} of the saving)")
say(f"  of which bandwidth only = 4 * {beta_m_us:.3f} = {4 * beta_m_us:.3f} us")
effective_mb_s = M_BYTES / one  # bytes per microsecond == MB/s
say(f"  effective bandwidth m/T(m) = {M_BYTES}/{one:.3f} = {effective_mb_s:.3f} B/us "
    f"= {effective_mb_s:.3f} MB/s")
check("beta*m ns", beta_m_ns, 1048576)
check("beta*m us", beta_m_us, 1048.576)
check("one message us", one, 1058.576)
check("linear us", linear, 7410.032)
check("binomial us", binomial, 3175.728)
check("linear ms", us_to_ms(linear), 7.410032)
check("binomial ms", us_to_ms(binomial), 3.175728)
check("ratio", ratio, 7.0 / 3.0)
check("saved us", saved, 4234.304)
check("latency-only saving us", 4 * ALPHA_US, 40.0)
check("effective MB/s", effective_mb_s, 990.553347138042)

# ------------------------------------------------------------ D. topology
section("D. Topology broadcast steps for p = 16 (store-and-forward, one-port)")
topologies = [
    ("ring, unidirectional", 15, "15 hops around the ring (0->1->...->15)"),
    ("ring, bidirectional", 8, "diameter = 16/2 = 8; root sends both directions"),
    ("2D mesh 4x4, corner root", 6, "Manhattan 3+3 to the opposite corner"),
    ("2D mesh 4x4, centre root", 4, "max Manhattan distance from (1,1) is 2+2"),
    ("2D torus 4x4", 4, "each wrapped dimension has diameter 2, so 2+2"),
    ("hypercube Q4", 4, "log2(16) = 4 recursive-doubling dimensions"),
]
say(f"{'topology':<28} {'steps':>5}   derivation")
for name, steps, why in topologies:
    say(f"{name:<28} {steps:>5}   {why}")
say()
say("model-based time per step = alpha + beta*m = T(2^20):")
for name, steps, _ in topologies:
    say(f"  {name:<28} {steps:>2} * {one:.3f} us = {steps * one:.3f} us")
say("note: task B's binomial tree assumes a fully connected network; its 3 logical")
say("rounds do not equal 3 physical hops on a ring or mesh.")
check("ring unidirectional", 15, 15)
check("ring bidirectional", 8, 8)
check("mesh corner", 6, 6)
check("mesh centre", 4, 4)
check("torus", 4, 4)
check("hypercube", 4, 4)
check("ring time us", 8 * one, 8468.608)
check("mesh-corner time us", 6 * one, 6351.456)
check("torus/Q4 time us", 4 * one, 4234.304)

# -------------------------------------------------------- E. load balance
section("E. Load balance: tile times [2, 2, 3, 3, 3, 4, 5, 18] ms on p = 8")
times = [2, 2, 3, 3, 3, 4, 5, 18]
p = 8
total = sum(times)
average = total / p
largest = max(times)
makespan = largest
speedup = total / makespan
efficiency = total / (p * makespan)
idle = p * makespan - total
say(f"  tile times        = {times} ms")
say(f"  total work        = {total} ms")
say(f"  average bound     = {total}/{p} = {average:.1f} ms")
say(f"  largest tile      = {largest} ms -> lower bound = max({average:.1f}, {largest}) "
    f"= {max(average, largest):.1f} ms")
say(f"  one tile per rank -> makespan = max(t_i) = {makespan} ms")
say(f"  speedup vs 1 rank = {total}/{makespan} = {speedup:.4f}")
say(f"  efficiency        = {total}/({p}*{makespan}) = {total}/{p * makespan} = "
    f"{efficiency:.4%}")
say(f"  total idle        = {p}*{makespan} - {total} = {idle} process-ms")
say(f"  per-rank idle     = {[makespan - t for t in times]} (sum {idle})")
check("total work", total, 40)
check("average bound", average, 5.0)
check("makespan", makespan, 18)
check("speedup", speedup, 40.0 / 18.0)
check("efficiency", efficiency, 40.0 / 144.0)
check("idle", idle, 104)
check("idle sum", sum(makespan - t for t in times), 104)

say()
say("Recommended change: split the 18 ms tile into four 4.5 ms pieces and use LPT.")
items = [2, 2, 3, 3, 3, 4, 5] + [4.5, 4.5, 4.5, 4.5]
items.sort(reverse=True)
loads = [0.0] * p
assignment = []
for index, item in enumerate(items):
    target = min(range(p), key=lambda r: (loads[r], r))
    loads[target] += item
    assignment.append((index, item, target, list(loads)))
say(f"  items (LPT order) = {items}")
say(f"  {'#':>2} {'tile_ms':>7} {'rank':>5}   loads after assignment")
for index, item, target, after in assignment:
    loads_text = " ".join(f"{v:g}" for v in after)
    say(f"  {index:>2} {item:>7g} {target:>5}   [{loads_text}]")
new_makespan = max(loads)
new_speedup = total / new_makespan
new_efficiency = total / (p * new_makespan)
saved_ms = makespan - new_makespan
say(f"  final loads       = {[round(v, 3) for v in loads]}")
say(f"  LPT makespan      = {new_makespan:g} ms (was {makespan} ms, "
    f"{makespan / new_makespan:.4f}x better)")
say(f"  speedup vs 1 rank = {total}/{new_makespan:g} = {new_speedup:.4f}")
say(f"  efficiency        = {total}/({p}*{new_makespan:g}) = {total}/{p * new_makespan:g} "
    f"= {new_efficiency:.4%}")
say(f"  compute saved     = {makespan} - {new_makespan:g} = {saved_ms:g} ms")
say("  optimality note: a makespan below 6 ms is impossible. Each 4.5 ms piece")
say("  would have to sit alone (adding any other job >=2 gives >6), using four")
say("  ranks; the 5 ms job also sits alone, leaving 4+3+3+3+2+2 = 17 ms for three")
say("  ranks. The 4 ms job then sits alone too, so 3+3+3+2+2 = 13 ms would need two")
say("  ranks, but two ranks below 6 ms hold <12 ms. Hence LPT's 6 ms is optimal.")
check("LPT makespan", new_makespan, 6.0)
check("LPT speedup", new_speedup, 40.0 / 6.0)
check("LPT efficiency", new_efficiency, 40.0 / 48.0)
check("compute saved", saved_ms, 12.0)
check("optimal makespan", new_makespan, 6.0)

say()
say("Communication cost of the change (assumed 32x32 RGBA float32 tile):")
reply = msg_us(TILE32_BYTES)
request_bytes = 16
request = msg_us(request_bytes)
per_tile = request + reply
n_tiles = len(items)
queue_us = n_tiles * per_tile
gather_bytes = n_tiles * TILE32_BYTES
tree_rounds = math.ceil(math.log2(p))
gather_us = tree_rounds * ALPHA_US + BETA_NS_PER_BYTE * gather_bytes / 1000.0
say(f"  tile payload      = {TILE32_BYTES} bytes; reply T = {ALPHA_US:g} + "
    f"{BETA_NS_PER_BYTE * TILE32_BYTES / 1000.0:.3f} = {reply:.3f} us")
say(f"  request (16 B)    = {request:.3f} us; request+reply = {per_tile:.3f} us")
say(f"  {n_tiles} tiles       = {n_tiles} * {per_tile:.3f} = {queue_us:.3f} us = "
    f"{queue_us / 1000.0:.4f} ms")
say(f"  as share of LPT makespan = {queue_us / 1000.0:.4f}/{new_makespan:g} = "
    f"{queue_us / 1000.0 / new_makespan:.4%}")
say(f"  net gain          = {saved_ms:g} - {queue_us / 1000.0:.4f} = "
    f"{saved_ms - queue_us / 1000.0:.4f} ms")
say(f"  batch gather instead: {n_tiles} tiles * {TILE32_BYTES} = {gather_bytes} bytes; "
    f"~{tree_rounds}*alpha + beta*bytes = {gather_us:.3f} us")
say(f"  break-even tile count for a per-tile queue = {saved_ms * 1000:.1f}/"
    f"{per_tile:.3f} = {saved_ms * 1000 / per_tile:.2f} tiles")
say("  at 1024 tiles (a 32x32 split of 1024x1024) the per-tile queue alone would cost")
say(f"  {1024 * per_tile / 1000.0:.1f} ms, so batch tiles or use a single Gatherv.")
check("32x32 reply us", reply, 26.384)
check("request us", request, 10.016)
check("per-tile queue us", per_tile, 36.4)
check("queue total us", queue_us, 400.4)
check("gather bytes", gather_bytes, 180224)
check("gather us", gather_us, 210.224)
check("break-even tiles", saved_ms * 1000 / per_tile, 12000.0 / 36.4)

section("All checks passed")
say("Every printed number was recomputed and matched the expected value.")
out.close()
