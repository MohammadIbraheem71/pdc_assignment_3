#!/usr/bin/env python3
"""Draw the two Part 5 diagrams with Pillow.

Outputs (next to this script):
    part5_dataflow.png     MPI data flow: broadcast -> tiles -> gather
    part5_topologies.png   p=16 ring, 4x4 mesh and Q4 hypercube with steps

matplotlib and graphviz are not installed on this machine, so the geometry is
explicit. Nothing here is measured; the numbers are the analytical values from
verify_part5.py.

Run:
    python3 part5/draw_part5_diagrams.py
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


BG = (250, 250, 252)
INK = (27, 32, 42)
MUTED = (92, 102, 118)
GRID = (190, 198, 212)
BLUE = (40, 100, 190)
GREEN = (30, 130, 80)
ORANGE = (200, 110, 30)
PURPLE = (110, 70, 160)
RED = (185, 60, 75)
LIGHT_BLUE = (226, 238, 252)
LIGHT_GREEN = (226, 245, 233)
LIGHT_ORANGE = (255, 242, 225)
LIGHT_PURPLE = (240, 234, 250)
WHITE = (255, 255, 255)


def lines(draw, cx, cy, text_lines, size, bold_first=False, fill=INK, gap=5):
    """Draw centered multi-line text; first line may be bold."""
    rendered = []
    for i, text in enumerate(text_lines):
        f = font(size, bold=(bold_first and i == 0))
        box = draw.textbbox((0, 0), text, font=f)
        rendered.append((text, f, box[3] - box[1]))
    total = sum(h for _, _, h in rendered) + gap * (len(rendered) - 1)
    y = cy - total / 2.0
    for text, f, h in rendered:
        draw.text((cx, y + h / 2.0), text, font=f, fill=fill, anchor="mm")
        y += h + gap


def box(draw, rect, fill, outline, width=2, radius=12):
    draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, points, color, width=3, head=13):
    for p1, p2 in zip(points, points[1:]):
        draw.line([p1, p2], fill=color, width=width, joint="curve")
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    left = (x2 - head * math.cos(angle - math.pi / 6),
            y2 - head * math.sin(angle - math.pi / 6))
    right = (x2 - head * math.cos(angle + math.pi / 6),
             y2 - head * math.sin(angle + math.pi / 6))
    draw.polygon([(x2, y2), left, right], fill=color)


def badge(draw, cx, cy, text, fill=WHITE, outline=GRID, radius=11, size=13, color=INK):
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                 fill=fill, outline=outline, width=1)
    draw.text((cx, cy), text, font=font(size, bold=True), fill=color, anchor="mm")


# --------------------------------------------------------------- data flow
def dataflow():
    width, height = 1640, 1140
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    draw.text((width / 2, 34), "Part 5: MPI data flow for the tiled renderer",
              font=font(32, bold=True), fill=INK, anchor="mm")
    draw.text((width / 2, 78),
              "Root broadcasts the scene; every rank renders only its own tiles; "
              "root gathers tile pixels in tile order and reassembles the image.",
              font=font(17), fill=MUTED, anchor="mm")
    draw.text((width / 2, 106),
              "Model-based design, no MPI code and nothing measured. p ranks in total; "
              "the worker pipeline is shown once because every rank runs it.",
              font=font(15), fill=MUTED, anchor="mm")

    # Root column.
    draw.text((245, 142), "ROOT rank 0", font=font(19, bold=True), fill=BLUE, anchor="mm")
    root_boxes = [
        ((50, 165, 440, 265), LIGHT_BLUE, BLUE,
         ["load scene", "16 B metadata + Nc x 28 B circles"]),
        ((50, 310, 440, 425), LIGHT_BLUE, BLUE,
         ["MPI_Bcast: metadata + full circles", "16 B + Nc x 28 B",
          "Nc=2000 -> 56,000 B"]),
        ((50, 470, 440, 590), LIGHT_BLUE, BLUE,
         ["assign tile ownership", "rank r takes tiles r, r+p, ...",
          "or MPI_Scatterv(counts, displs)"]),
        ((50, 635, 440, 750), LIGHT_BLUE, BLUE,
         ["root allocates final image", "1024x1024 RGBA float32 = 16 MiB",
          "white fill before the gather"]),
    ]
    for rect, fill, outline, text in root_boxes:
        box(draw, rect, fill, outline)
        lines(draw, (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2, text, 16,
              bold_first=True, gap=4)

    # Worker column.
    draw.text((800, 142), "EVERY WORKER rank r (same code on all p ranks)",
              font=font(19, bold=True), fill=GREEN, anchor="mm")
    worker_boxes = [
        ((585, 165, 1015, 255), ["receive metadata + circle array", "same message on every rank"]),
        ((585, 280, 1015, 370), ["obtain owned tile IDs", "rank formula or MPI_Scatterv"]),
        ((585, 395, 1015, 485), ["build per-tile CSR candidate lists", "padded box -> count -> prefix -> fill"]),
        ((585, 510, 1015, 600), ["render owned tiles locally", "covers()/blend(), ascending circle order"]),
        ((585, 625, 1015, 715), ["pack tile pixels in tile order", "send tile buffers to root"]),
    ]
    for rect, text in worker_boxes:
        box(draw, rect, LIGHT_GREEN, GREEN)
        lines(draw, (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2, text, 16,
              bold_first=True, gap=4)

    # Right column: gather and reassembly.
    draw.text((1345, 142), "ROOT rank 0", font=font(19, bold=True), fill=ORANGE, anchor="mm")
    gather_boxes = [
        ((1100, 165, 1590, 320), LIGHT_ORANGE, ORANGE,
         ["MPI_Gatherv onto root", "counts[] = tiles per rank",
          "displs[] = tile-order offsets", "variable counts"]),
        ((1100, 360, 1590, 475), LIGHT_ORANGE, ORANGE,
         ["reassemble final image", "copy each returned tile into", "its (tx, ty) position"]),
        ((1100, 515, 1590, 610), LIGHT_ORANGE, ORANGE,
         ["write output PPM/PNG", "only after the full gather"]),
    ]
    for rect, fill, outline, text in gather_boxes:
        box(draw, rect, fill, outline)
        lines(draw, (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2, text, 16,
              bold_first=True, gap=4)

    # Root pipeline arrows.
    arrow(draw, [(245, 265), (245, 310)], BLUE)
    arrow(draw, [(245, 425), (245, 470)], BLUE)
    arrow(draw, [(245, 590), (245, 635)], BLUE)

    # Broadcast fan-out.
    arrow(draw, [(440, 345), (512, 345), (512, 210), (585, 210)], GREEN, width=4)
    draw.text((515, 285), "MPI_Bcast", font=font(15, bold=True), fill=GREEN)
    draw.text((515, 303), "one m-byte message", font=font(13), fill=MUTED)

    # Tile assignment.
    arrow(draw, [(440, 530), (545, 530), (545, 325), (585, 325)], GREEN, width=4)
    draw.text((548, 392), "tile IDs", font=font(15, bold=True), fill=GREEN)
    draw.text((548, 410), "Scatterv or formula", font=font(13), fill=MUTED)

    # Worker pipeline.
    worker_rects = [rect for rect, _ in worker_boxes]
    for (x0, y0, x1, y1), (_, y2, _, _) in zip(worker_rects, worker_rects[1:]):
        arrow(draw, [((x0 + x1) / 2, y1), ((x0 + x1) / 2, y2)], GREEN)

    # Worker send to gather.
    arrow(draw, [(1015, 670), (1060, 670), (1060, 250), (1100, 250)], ORANGE, width=4)
    draw.text((1065, 465), "tile buffers", font=font(14, bold=True), fill=ORANGE)
    draw.text((1065, 483), "in tile order", font=font(13), fill=MUTED)

    # Root allocation is ready before reassembly.
    arrow(draw, [(440, 692), (500, 692), (500, 860), (1045, 860), (1045, 418),
                 (1100, 418)], ORANGE, width=3)
    draw.text((748, 844), "image already allocated on root", font=font(14), fill=MUTED)

    # Gather -> reassemble -> write.
    arrow(draw, [(1345, 320), (1345, 360)], ORANGE)
    arrow(draw, [(1345, 475), (1345, 515)], ORANGE)

    # Bottom explanation.
    box(draw, (50, 900, 1590, 1110), WHITE, GRID, width=2)
    draw.text((72, 922), "Why the variable-count collectives, and what crosses the network",
              font=font(19, bold=True), fill=INK)
    bullets = [
        "Fixed-count MPI_Scatter / MPI_Gather require identical counts on every rank. "
        "Tiles are not divisible by p and edge tiles are smaller when the image is not divisible by the tile size.",
        "Example from verify_part5.py: 1050x1050 image, 16-pixel tiles -> 4356 tiles; p=8 -> "
        "counts=[545,545,545,545,544,544,544,544], displs=[0,545,1090,1635,2180,2724,3268,3812].",
        "Tiles are non-contiguous in the final image. Chosen: gather tile buffers in tile order and "
        "reassemble on root. No MPI_Type_create_subarray/pack needed, and the code stays simple.",
        "Recommended: broadcast the full circle array (28 B x Nc = 56,000 B at Nc=2000) because it is "
        "one small message; per-rank overlap-only sends save bytes but add root filtering and many messages.",
    ]
    y = 958
    for text in bullets:
        draw.text((74, y), "-", font=font(16, bold=True), fill=BLUE)
        # Wrap at a conservative character count for this font.
        words, line, wrapped = text.split(), "", []
        for word in words:
            trial = word if not line else line + " " + word
            if draw.textlength(trial, font=font(15)) > 1470:
                wrapped.append(line)
                line = word
            else:
                line = trial
        if line:
            wrapped.append(line)
        for part in wrapped:
            draw.text((96, y), part, font=font(15), fill=INK)
            y += 24
        y += 5

    image.save(HERE / "part5_dataflow.png")
    print("wrote", HERE / "part5_dataflow.png")


# --------------------------------------------------------------- topologies
def step_color(step, max_step=8):
    if step == 0:
        return GREEN
    t = (step - 1) / max(1, max_step - 1)
    return (int(245 - 195 * t), int(165 - 110 * t), int(60 + 100 * t))


def ring_panel(draw, x0, title):
    draw.text((x0 + 300, 155), title, font=font(22, bold=True), fill=INK, anchor="mm")
    draw.text((x0 + 300, 183), "diameter = 16/2 = 8 steps; root sends both ways in round 1",
              font=font(14), fill=MUTED, anchor="mm")
    cx, cy = x0 + 300, 555
    rx, ry = 205, 290
    positions = []
    for i in range(16):
        angle = -math.pi / 2 + 2 * math.pi * i / 16
        positions.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))

    # Ring edges, colored by the broadcast round that uses them.
    for i in range(16):
        j = (i + 1) % 16
        # Clockwise path from 0: edge 0-1 is round 1 ... edge 7-8 is round 8.
        # Counter-clockwise path from 0: edge 15-0 is round 1 ... edge 9-8 is round 8.
        if i <= 7:
            rnd = i + 1
            p1, p2 = positions[i], positions[j]
        else:
            rnd = 16 - i
            p1, p2 = positions[j], positions[i]
        color = step_color(rnd)
        draw.line([p1, p2], fill=color, width=4)
        # Arrowhead near the destination (boxes are drawn afterwards).
        mx, my = p1[0] + 0.58 * (p2[0] - p1[0]), p1[1] + 0.58 * (p2[1] - p1[1])
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        head = 10
        draw.polygon([(mx, my),
                      (mx - head * math.cos(angle - math.pi / 6),
                       my - head * math.sin(angle - math.pi / 6)),
                      (mx - head * math.cos(angle + math.pi / 6),
                       my - head * math.sin(angle + math.pi / 6))], fill=color)
        # Edge round badge at the midpoint.
        badge(draw, (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, str(rnd),
              fill=WHITE, outline=color, radius=10, size=12, color=color)

    for i, (px, py) in enumerate(positions):
        step = min(i, 16 - i)
        color = step_color(step)
        box(draw, (px - 32, py - 25, px + 32, py + 25), color, color, width=2, radius=9)
        draw.text((px, py - 7), f"r{i}", font=font(14, bold=True), fill=WHITE, anchor="mm")
        draw.text((px, py + 11), f"s{step}", font=font(13), fill=WHITE, anchor="mm")

    draw.text((x0 + 300, 890),
              "node labels: rN = rank, sN = round reached; edge badge = round used",
              font=font(14), fill=MUTED, anchor="mm")
    draw.text((x0 + 300, 912),
              "all 16 edges carry one message; node 8 is reached in round 8 from both sides",
              font=font(14), fill=MUTED, anchor="mm")


def mesh_panel(draw, x0, title):
    draw.text((x0 + 300, 155), title, font=font(22, bold=True), fill=INK, anchor="mm")
    draw.text((x0 + 300, 183), "root at corner; s = Manhattan distance; diameter = 3+3 = 6",
              font=font(14), fill=MUTED, anchor="mm")
    xs = [x0 + 80 + 135 * i for i in range(4)]
    ys = [285 + 135 * j for j in range(4)]
    positions = {}
    for ty in range(4):
        for tx in range(4):
            rank = ty * 4 + tx
            positions[rank] = (xs[tx], ys[ty])

    # All grid links, arrowheads point away from the root (right/down).
    for ty in range(4):
        for tx in range(4):
            rank = ty * 4 + tx
            p = positions[rank]
            if tx < 3:
                q = positions[rank + 1]
                color = step_color(tx + ty + 1)
                draw.line([p, q], fill=color, width=4)
                mx, my = p[0] + 0.56 * (q[0] - p[0]), p[1] + 0.56 * (q[1] - p[1])
                draw.polygon([(mx, my), (mx - 10, my - 7), (mx - 10, my + 7)], fill=color)
            if ty < 3:
                q = positions[rank + 4]
                color = step_color(tx + ty + 1)
                draw.line([p, q], fill=color, width=4)
                mx, my = p[0] + 0.56 * (q[0] - p[0]), p[1] + 0.56 * (q[1] - p[1])
                draw.polygon([(mx, my), (mx - 7, my - 10), (mx + 7, my - 10)], fill=color)

    for rank, (px, py) in positions.items():
        tx, ty = rank % 4, rank // 4
        step = tx + ty
        color = step_color(step)
        box(draw, (px - 33, py - 24, px + 33, py + 24), color, color, width=2, radius=8)
        draw.text((px, py - 7), f"r{rank}", font=font(14, bold=True), fill=WHITE, anchor="mm")
        draw.text((px, py + 11), f"s{step}", font=font(13), fill=WHITE, anchor="mm")

    draw.text((x0 + 300, 880),
              "arrows: outward store-and-forward wave; sN is the round the node is reached",
              font=font(13), fill=MUTED, anchor="mm")
    draw.text((x0 + 300, 905),
              "root at centre: max distance = 2+2 = 4 steps",
              font=font(14), fill=INK, anchor="mm")
    draw.text((x0 + 300, 928),
              "4x4 torus (wrapped edges): 2+2 = 4 steps",
              font=font(14), fill=INK, anchor="mm")


def hypercube_panel(draw, x0, title):
    draw.text((x0 + 300, 155), title, font=font(22, bold=True), fill=INK, anchor="mm")
    draw.text((x0 + 300, 183), "two Q3 cubes joined by bit b3; step = popcount(rank xor 0000)",
              font=font(14), fill=MUTED, anchor="mm")

    # Binary rank -> position. b2 is the depth bit, b1 vertical, b0 horizontal.
    base_x = x0 + 55
    positions = {}
    for high in (0, 1):
        front_x = base_x + high * 270
        for b2 in (0, 1):
            for b1 in (0, 1):
                for b0 in (0, 1):
                    rank = (high << 3) | (b2 << 2) | (b1 << 1) | b0
                    px = front_x + b0 * 130 + b2 * 70
                    py = 635 - b1 * 145 - b2 * 65
                    positions[rank] = (px, py)

    bit_colors = {0: RED, 1: GREEN, 2: BLUE, 3: PURPLE}
    edges = set()

    def add_edge(a, b):
        if (a, b) not in edges and (b, a) not in edges:
            edges.add((min(a, b), max(a, b)))

    for high in (0, 1):
        base = high * 8
        # Q3 cube: four front edges, four back edges, four depth edges.
        for a, b in [(0, 1), (0, 2), (1, 3), (2, 3),
                     (4, 5), (4, 6), (5, 7), (6, 7),
                     (0, 4), (1, 5), (2, 6), (3, 7)]:
            add_edge(base + a, base + b)
    for a in range(8):
        add_edge(a, a + 8)

    for a, b in sorted(edges):
        diff = a ^ b
        bit = (diff.bit_length() - 1)
        color = bit_colors[bit]
        p, q = positions[a], positions[b]
        draw.line([p, q], fill=color, width=4)
        # Direction: smaller rank -> larger rank just for an arrowhead marker.
        mx, my = p[0] + 0.56 * (q[0] - p[0]), p[1] + 0.56 * (q[1] - p[1])
        angle = math.atan2(q[1] - p[1], q[0] - p[0])
        head = 9
        draw.polygon([(mx, my),
                      (mx - head * math.cos(angle - math.pi / 6),
                       my - head * math.sin(angle - math.pi / 6)),
                      (mx - head * math.cos(angle + math.pi / 6),
                       my - head * math.sin(angle + math.pi / 6))], fill=color)

    for rank, (px, py) in positions.items():
        step = bin(rank).count("1")
        color = step_color(step, max_step=4)
        box(draw, (px - 35, py - 23, px + 35, py + 23), color, color, width=2, radius=8)
        draw.text((px, py - 7), format(rank, "04b"), font=font(13, bold=True), fill=WHITE,
                  anchor="mm")
        draw.text((px, py + 11), f"s{step}", font=font(12), fill=WHITE, anchor="mm")

    draw.text((x0 + 300, 850),
              "edge colour = dimension fixed in that round: b0 round 1, b1 round 2,",
              font=font(13), fill=MUTED, anchor="mm")
    draw.text((x0 + 300, 871),
              "b2 round 3, b3 round 4. All 16 nodes are reached in 4 rounds.",
              font=font(13), fill=MUTED, anchor="mm")
    draw.text((x0 + 300, 902),
              "left cube = high bit b3=0; right cube = b3=1; 32 bidirectional links",
              font=font(13), fill=MUTED, anchor="mm")
    draw.text((x0 + 300, 930),
              "steps = log2(16) = 4 = diameter(Q4)",
              font=font(14, bold=True), fill=INK, anchor="mm")


def topologies():
    width, height = 1900, 1270
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    draw.text((width / 2, 34), "Part 5: broadcast topologies for p = 16, root = rank 0",
              font=font(32, bold=True), fill=INK, anchor="mm")
    draw.text((width / 2, 78),
              "Store-and-forward step counting: one step = one round of simultaneous sends along direct links.",
              font=font(17), fill=MUTED, anchor="mm")
    draw.text((width / 2, 104),
              "All links bidirectional; model-based only, nothing measured. Strict one-port serialization is discussed in the notes.",
              font=font(15), fill=MUTED, anchor="mm")

    panels = [
        (30, 620, "Ring (16 nodes, bidirectional)"),
        (650, 1240, "2D mesh 4x4 (no wraparound)"),
        (1270, 1860, "Hypercube Q4 (dimension 4)"),
    ]
    for x1, x2, title in panels:
        draw.rounded_rectangle((x1, 125, x2, 950), radius=18, fill=WHITE, outline=GRID, width=2)
        if title.startswith("Ring"):
            ring_panel(draw, x1, title)
        elif title.startswith("2D"):
            mesh_panel(draw, x1, title)
        else:
            hypercube_panel(draw, x1, title)

    # Bottom legend and caveat.
    draw.rounded_rectangle((30, 970, 1860, 1240), radius=16, fill=WHITE, outline=GRID, width=2)
    draw.text((58, 992), "Reading the step labels", font=font(19, bold=True), fill=INK)
    legend = [
        ("green node", "root, s0"),
        ("ring / mesh", "sN = round reached; edge badge = round used"),
        ("hypercube", "sN = Hamming weight from 0000; edge colour = bit fixed"),
        ("ring", "8 steps = ceil(16/2), not 15"),
        ("mesh", "6 steps from a corner; 4 from the centre or on a torus"),
        ("Q4", "4 steps = log2(16)"),
    ]
    for index, (label, text) in enumerate(legend):
        x = 58 + (index % 3) * 610
        y = 1035 + (index // 3) * 72
        draw.text((x, y), label + ":", font=font(14, bold=True), fill=BLUE)
        draw.text((x, y + 22), text, font=font(14), fill=INK)
    draw.text((58, 1188),
              "Step counts ignore beta*m, link contention, bisection bandwidth, link speeds, rank-to-hardware mapping, "
              "pipelining and software overhead; see section D of part5_notes.md.",
              font=font(14), fill=MUTED)
    draw.text((58, 1211),
              "Real clusters are usually switched fat-trees, not rings or meshes; a physical/logical topology mismatch can dominate these counts.",
              font=font(14), fill=MUTED)

    image.save(HERE / "part5_topologies.png")
    print("wrote", HERE / "part5_topologies.png")


if __name__ == "__main__":
    dataflow()
    topologies()
