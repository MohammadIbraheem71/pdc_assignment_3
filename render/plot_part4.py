#!/usr/bin/env python3
"""Part 4 graphs from the measured CSV files.

Usage (from the repository root, after the benchmarks):
    python3 render/plot_part4.py

Writes:
    part4_speedup_vs_threads.png  baseline vs optimized speedup per scene
    part4_tile_schedule.png       runtime vs tile size and schedule
    part4_busy_threads.png        per-thread busy time, static vs dynamic

matplotlib is used when installed; otherwise the script falls back to Pillow so
the PNGs are still produced from the same CSV data. Every axis is labeled and
each image ends with a one-line note explaining what it shows.
"""
import csv
import math
import os
import sys

MAIN_CSV = "part4_results_dynamic.csv"
SWEEP_CSV = "part4_sweep.csv"
BUSY_CSV = "part4_sweep_busy.csv"

SCENES = ["p4_sparse", "p4_heavy", "p4_corner", "p4_few", "p4_one", "p4_huge"]
SCENE_TITLES = {
    "p4_sparse": "p4_sparse (2000 tiny)",
    "p4_heavy": "p4_heavy (400 large)",
    "p4_corner": "p4_corner (clustered)",
    "p4_few": "p4_few (8 circles)",
    "p4_one": "p4_one (1 huge)",
    "p4_huge": "p4_huge (32 huge)",
}
SCHEDULE_ORDER = ["static", "dynamic,1", "dynamic,4", "dynamic,16", "guided,4"]
SCHEDULE_COLORS = {
    "static": "#1f77b4",
    "dynamic,1": "#d95f02",
    "dynamic,4": "#2ca02c",
    "dynamic,16": "#9467bd",
    "guided,4": "#8c564b",
}
BASELINE_COLOR = "#1f77b4"
OPTIMIZED_COLOR = "#d95f02"
NOTE_SPEEDUP = (
    "Speedup Sp = T1/Tp with T1 = baseline render_parallel on 1 thread per scene; "
    "optimized = tile 16, dynamic chunk 4; log scale."
)
NOTE_TILE = (
    "Median of 5 runs at 8 threads, 1024x1024: dynamic schedules fix the clustered-scene "
    "imbalance; tile 8 pays high preprocessing on p4_heavy."
)
NOTE_BUSY = (
    "Measured tile-loop time per worker at tile 16, 8 threads: static block scheduling "
    "leaves most workers idle on p4_corner; dynamic chunk 4 balances both scenes."
)


def read_csv(path):
    if not os.path.exists(path):
        sys.exit(f"error: {path} not found (run the Part 4 benchmarks first)")
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def speedup_series():
    rows = read_csv(MAIN_CSV)
    series = {}
    for row in rows:
        scene = row["scene"]
        series.setdefault(scene, {}).setdefault(row["variant"], []).append(
            (int(row["threads"]), float(row["speedup"]))
        )
    for variants in series.values():
        for points in variants.values():
            points.sort()
    return series


def sweep_series():
    rows = read_csv(SWEEP_CSV)
    series = {}
    for row in rows:
        if float(row["threads"] if "threads" in row else 8) < 1:
            continue
        key = (row["scene"], row["schedule"])
        series.setdefault(key, []).append((int(row["tile"]), float(row["median_ms"])))
    for points in series.values():
        points.sort()
    return series


def busy_series():
    rows = read_csv(BUSY_CSV)
    series = {}
    for row in rows:
        if int(row["tile"]) != 16:
            continue
        key = (row["scene"], row["schedule"])
        series.setdefault(key, []).append((int(row["thread"]), float(row["busy_ms"])))
    for points in series.values():
        points.sort()
    return series


def plain(value):
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


# ---------------------------------------------------------------- matplotlib


def plot_with_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    fig = plt.figure(figsize=(12, 9), dpi=140)
    grid = gridspec.GridSpec(3, 2, hspace=0.55, wspace=0.25)

    series = speedup_series()
    for i, scene in enumerate(SCENES):
        ax = fig.add_subplot(grid[i // 2, i % 2])
        for variant, color, marker in (("baseline", BASELINE_COLOR, "o"),
                                       ("optimized", OPTIMIZED_COLOR, "s")):
            points = series.get(scene, {}).get(variant, [])
            ax.plot([p for p, _ in points], [s for _, s in points], marker=marker,
                    linewidth=2, color=color, label=variant)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_yscale("log")
        ax.set_xlim(0.9, 8.8)
        ax.set_title(SCENE_TITLES[scene], fontsize=10)
        ax.set_xlabel("Threads p", fontsize=9)
        ax.set_ylabel("Speedup Sp (log)", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
    fig.suptitle("Part 4: baseline vs optimized renderer, speedup vs threads\n" +
                 NOTE_SPEEDUP, fontsize=10)
    fig.savefig("part4_speedup_vs_threads.png", bbox_inches="tight")
    plt.close(fig)

    # Tile/schedule.
    sweep = sweep_series()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)
    for ax, scene in zip(axes, ["p4_corner", "p4_heavy"]):
        for schedule in SCHEDULE_ORDER:
            points = sweep.get((scene, schedule), [])
            if not points:
                continue
            ax.plot([t for t, _ in points], [m for _, m in points], marker="o",
                    linewidth=2, color=SCHEDULE_COLORS[schedule], label=schedule)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64])
        ax.set_xticklabels(["8", "16", "32", "64"])
        ax.set_xlabel("Tile size (pixels)")
        ax.set_ylabel("Median total time (ms)")
        ax.set_title(SCENE_TITLES[scene], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Part 4: runtime vs tile size and schedule\n" + NOTE_TILE, fontsize=10)
    fig.savefig("part4_tile_schedule.png", bbox_inches="tight")
    plt.close(fig)

    # Per-thread busy time.
    busy = busy_series()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)
    for ax, scene in zip(axes, ["p4_corner", "p4_heavy"]):
        static = busy.get((scene, "static"), [])
        dynamic = busy.get((scene, "dynamic,4"), [])
        xs = [t for t, _ in static]
        width = 0.4
        ax.bar([x - width / 2 for x in xs], [b for _, b in static], width,
               color=BASELINE_COLOR, label="static")
        ax.bar([x + width / 2 for x in xs], [b for _, b in dynamic], width,
               color=OPTIMIZED_COLOR, label="dynamic,4")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel("Worker thread")
        ax.set_ylabel("Busy time in tiles (ms)")
        ax.set_title(SCENE_TITLES[scene], fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Part 4: per-thread busy time at tile 16, 8 threads\n" + NOTE_BUSY,
                 fontsize=10)
    fig.savefig("part4_busy_threads.png", bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------- Pillow


class Panel:
    def __init__(self, draw, rect, title, xlabel, ylabel, font, small):
        self.draw = draw
        self.x0, self.y0, self.x1, self.y1 = rect
        self.font = font
        self.small = small
        draw.rectangle((self.x0, self.y0, self.x1, self.y1), outline="black", width=2)
        draw.text(((self.x0 + self.x1) / 2, self.y0 - 14), title, fill="black",
                  font=small, anchor="mb")
        draw.text(((self.x0 + self.x1) / 2, self.y1 + 46), xlabel, fill="black",
                  font=font, anchor="mm")
        # Horizontal y-axis label to the left of the panel (Pillow cannot rotate
        # text without extra images, so the full label is written sideways).
        draw.text((self.x0 - 100, (self.y0 + self.y1) / 2), ylabel, fill="black",
                  font=small, anchor="mm")

    def frame(self, xmin, xmax, ymin, ymax):
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax

    def x(self, value):
        span = self.xmax - self.xmin
        return self.x0 + (value - self.xmin) / span * (self.x1 - self.x0)

    def y(self, value):
        span = self.ymax - self.ymin
        return self.y1 - (value - self.ymin) / span * (self.y1 - self.y0)

    def ylog(self, value):
        low, high = math.log10(self.ymin), math.log10(self.ymax)
        frac = (math.log10(max(value, self.ymin)) - low) / (high - low)
        return self.y1 - frac * (self.y1 - self.y0)

    def grid_linear(self, values, fmt):
        for value in values:
            y = self.y(value)
            self.draw.line((self.x0, y, self.x1, y), fill=(225, 225, 225))
            self.draw.text((self.x0 - 10, y), fmt.format(value), fill="black",
                           font=self.font, anchor="rm")

    def grid_log(self, values, fmt):
        for value in values:
            y = self.ylog(value)
            self.draw.line((self.x0, y, self.x1, y), fill=(225, 225, 225))
            self.draw.text((self.x0 - 10, y), fmt.format(value), fill="black",
                           font=self.font, anchor="rm")

    def xticks(self, values, labels):
        for value, label in zip(values, labels):
            x = self.x(value)
            self.draw.line((x, self.y1, x, self.y1 + 6), fill="black", width=2)
            self.draw.text((x, self.y1 + 10), label, fill="black", font=self.font,
                           anchor="ma")


def pillow_canvas(title, note, panels, width=1400, height=1120):
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = bold = tiny = ImageFont.load_default()
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width / 2, 30), title, fill="black", font=bold, anchor="mm")
    return image, draw, font, tiny


def plot_with_pillow():
    from PIL import ImageDraw

    # 1. Speedup vs threads, 2x3 small multiples.
    series = speedup_series()
    image, draw, font, tiny = pillow_canvas(
        "Part 4: baseline vs optimized renderer, speedup vs threads",
        NOTE_SPEEDUP, None)
    left, top = 210, 130
    panel_w, panel_h = 480, 230
    gap_x, gap_y = 90, 100
    ticks = [0.5, 1, 10, 100, 1000, 2000]
    for i, scene in enumerate(SCENES):
        px = left + (i % 2) * (panel_w + gap_x)
        py = top + (i // 2) * (panel_h + gap_y)
        panel = Panel(draw, (px, py, px + panel_w, py + panel_h), SCENE_TITLES[scene],
                      "Threads p", "Speedup Sp (log)", font, tiny)
        panel.frame(0.9, 8.8, 0.5, 2000.0)
        panel.grid_log(ticks, "{:g}")
        panel.xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        for variant, color in (("baseline", BASELINE_COLOR), ("optimized", OPTIMIZED_COLOR)):
            points = [(panel.x(p), panel.ylog(s))
                      for p, s in series.get(scene, {}).get(variant, [])]
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            for x, y in points:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white")
    legend_y = 82
    for x, label, color in ((860, "baseline", BASELINE_COLOR),
                            (1080, "optimized (tile 16, dynamic,4)", OPTIMIZED_COLOR)):
        draw.line((x, legend_y, x + 36, legend_y), fill=color, width=4)
        draw.text((x + 44, legend_y), label, fill="black", font=tiny, anchor="lm")
    draw.text((40, 1092), NOTE_SPEEDUP, fill=(60, 60, 60), font=tiny, anchor="lm")
    image.save("part4_speedup_vs_threads.png")

    # 2. Runtime vs tile size and schedule.
    sweep = sweep_series()
    image, draw, font, tiny = pillow_canvas(
        "Part 4: runtime vs tile size and schedule", NOTE_TILE, None,
        width=1400, height=1100)
    panel_w, panel_h = 480, 620
    for idx, scene in enumerate(["p4_corner", "p4_heavy"]):
        px = 210 + idx * (panel_w + 80)
        py = 160
        panel = Panel(draw, (px, py, px + panel_w, py + panel_h), SCENE_TITLES[scene],
                      "Tile size (pixels)", "Median total time (ms)", font, tiny)
        values = [m for (s, _), pts in sweep.items() if s == scene for _, m in pts]
        ymax = math.ceil(max(values) / 20.0) * 20.0
        panel.frame(0.0, 3.0, 0.0, ymax)
        panel.grid_linear([ymax * k / 5 for k in range(6)], "{:.0f}")
        panel.xticks([0, 1, 2, 3], ["8", "16", "32", "64"])
        for schedule in SCHEDULE_ORDER:
            points = [(panel.x(math.log2(t / 8.0)), panel.y(m))
                      for t, m in sweep.get((scene, schedule), [])]
            if len(points) > 1:
                draw.line(points, fill=SCHEDULE_COLORS[schedule], width=3)
            for x, y in points:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=SCHEDULE_COLORS[schedule],
                             outline="white")
    legend_x, legend_y = 300, 850
    for i, schedule in enumerate(SCHEDULE_ORDER):
        y = legend_y + (i % 3) * 28
        x = legend_x + (i // 3) * 300
        draw.line((x, y, x + 30, y), fill=SCHEDULE_COLORS[schedule], width=4)
        draw.text((x + 40, y), schedule, fill="black", font=tiny, anchor="lm")
    draw.text((40, 1060), NOTE_TILE, fill=(60, 60, 60), font=tiny, anchor="lm")
    image.save("part4_tile_schedule.png")

    # 3. Per-thread busy time.
    busy = busy_series()
    image, draw, font, tiny = pillow_canvas(
        "Part 4: per-thread busy time at tile 16, 8 threads", NOTE_BUSY, None,
        width=1400, height=980)
    panel_w, panel_h = 480, 600
    for idx, scene in enumerate(["p4_corner", "p4_heavy"]):
        px = 210 + idx * (panel_w + 80)
        py = 170
        panel = Panel(draw, (px, py, px + panel_w, py + panel_h), SCENE_TITLES[scene],
                      "Worker thread", "Busy time in tiles (ms)", font, tiny)
        static = busy.get((scene, "static"), [])
        dynamic = busy.get((scene, "dynamic,4"), [])
        ymax = math.ceil(max(b for _, b in static + dynamic) / 20.0) * 20.0
        panel.frame(-0.6, 7.6, 0.0, ymax)
        panel.grid_linear([ymax * k / 5 for k in range(6)], "{:.0f}")
        panel.xticks(list(range(8)), [str(t) for t in range(8)])
        for values, color, offset in ((static, BASELINE_COLOR, -0.18),
                                      (dynamic, OPTIMIZED_COLOR, 0.18)):
            for thread, value in values:
                x = panel.x(thread + offset)
                y = panel.y(value)
                half = 0.15 / 8.2 * panel_w
                draw.rectangle((x - half, y, x + half, panel.y(0.0)), fill=color)
    legend_y = 870
    for i, (label, color) in enumerate((("static", BASELINE_COLOR),
                                        ("dynamic,4", OPTIMIZED_COLOR))):
        x = 320 + i * 300
        draw.rectangle((x, legend_y - 8, x + 26, legend_y + 8), fill=color)
        draw.text((x + 36, legend_y), label, fill="black", font=tiny, anchor="lm")
    draw.text((40, 930), NOTE_BUSY, fill=(60, 60, 60), font=tiny, anchor="lm")
    image.save("part4_busy_threads.png")


def main():
    try:
        plot_with_matplotlib()
        backend = "matplotlib"
    except ImportError:
        plot_with_pillow()
        backend = "Pillow fallback (matplotlib not installed)"
    print(f"wrote part4_speedup_vs_threads.png, part4_tile_schedule.png, "
          f"part4_busy_threads.png using {backend}")


if __name__ == "__main__":
    main()
