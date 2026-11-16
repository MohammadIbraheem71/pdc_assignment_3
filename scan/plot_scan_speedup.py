#!/usr/bin/env python3
"""Plot Part 2 speedup vs threads from the benchmark CSV.

Usage:
    python3 scan/plot_scan_speedup.py [scan_results.csv] [scan_speedup.png]

Speedup is S_p = T1 / Tp, where T1 is the parallel function at one thread.
matplotlib is used when it is installed; otherwise the script falls back to
Pillow (PIL) so the PNG can still be produced from the measured CSV.
"""
import csv
import math
import os
import sys

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "scan_results.csv"
PNG_PATH = sys.argv[2] if len(sys.argv) > 2 else "scan_speedup.png"

# (size, op) -> (colour, dash pattern, label)
STYLE = {
    (100000, "scan"): ("#1f77b4", None, "n = 1e5, scan"),
    (100000, "repeats"): ("#1f77b4", (8, 6), "n = 1e5, repeats"),
    (10000000, "scan"): ("#d95f02", None, "n = 1e7, scan"),
    (10000000, "repeats"): ("#d95f02", (8, 6), "n = 1e7, repeats"),
}


def load_rows(path):
    series = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] != "parallel" or row["speedup"] == "":
                continue
            key = (int(row["n"]), row["op"])
            series.setdefault(key, []).append((int(row["threads"]), float(row["speedup"])))
    for points in series.values():
        points.sort()
    return series


def nice_ceiling(value, step=0.5):
    return math.ceil(value / step) * step


def plot_with_matplotlib(series, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for key in sorted(series):
        color, dash, label = STYLE.get(key, (None, None, f"n = {key[0]}, {key[1]}"))
        xs = [p for p, _ in series[key]]
        ys = [s for _, s in series[key]]
        line, = ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=label)
        if dash:
            line.set_dashes(dash)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="T1 baseline (S = 1)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlim(0.9, 8.8)
    top = max(s for points in series.values() for _, s in points)
    ax.set_ylim(0, nice_ceiling(top + 0.5))
    ax.set_xlabel("Threads p")
    ax.set_ylabel("Speedup $S_p = T_1 / T_p$")
    ax.set_title("Exclusive scan and repeat compaction speedup (measured medians)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_with_pillow(series, path):
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1200, 800
    left, right, top, bottom = 110, 60, 90, 110
    plot_w = width - left - right
    plot_h = height - top - bottom

    threads = sorted({p for points in series.values() for p, _ in points})
    max_speedup = max(s for points in series.values() for _, s in points)
    y_max = nice_ceiling(max(max_speedup + 0.5, 1.5))

    def x_of(p):
        return left + (math.log2(p) / math.log2(max(threads))) * plot_w

    def y_of(s):
        return top + plot_h - (s / y_max) * plot_h

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
        bold = font

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Horizontal grid lines and y tick labels.
    steps = int(y_max * 2)
    for i in range(steps + 1):
        value = y_max * i / steps
        y = y_of(value)
        draw.line((left, y, left + plot_w, y), fill=(220, 220, 220), width=1)
        draw.text((left - 12, y), f"{value:.1f}", fill="black", font=font, anchor="rm")

    # Axes.
    draw.line((left, top, left, top + plot_h), fill="black", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=2)

    # T1 baseline.
    y_one = y_of(1.0)
    for x in range(left, left + plot_w, 14):
        draw.line((x, y_one, min(x + 7, left + plot_w), y_one), fill=(120, 120, 120), width=2)
    draw.text((left + plot_w - 8, y_of(1.08)), "T1 baseline (S = 1)", fill=(90, 90, 90),
              font=font, anchor="rb")

    def dashed_line(points, color, dash, width_px=3):
        """Draw a dashed polyline segment by segment."""
        on, off = dash
        for index in range(len(points) - 1):
            (x0, y0), (x1, y1) = points[index], points[index + 1]
            length = math.hypot(x1 - x0, y1 - y0)
            travelled = 0.0
            drawing = True
            while travelled < length:
                run = (on if drawing else off) * 3.0
                start = travelled / length
                stop = min(1.0, (travelled + run) / length)
                if drawing:
                    draw.line(
                        (x0 + (x1 - x0) * start, y0 + (y1 - y0) * start,
                         x0 + (x1 - x0) * stop, y0 + (y1 - y0) * stop),
                        fill=color, width=width_px)
                travelled += run
                drawing = not drawing

    for key in sorted(series):
        color, dash, _ = STYLE.get(key, ("#333333", None, ""))
        points = [(x_of(p), y_of(s)) for p, s in series[key]]
        if len(points) > 1:
            if dash:
                dashed_line(points, color, dash)
            else:
                draw.line(points, fill=color, width=3, joint="curve")
        for x, y in points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="white", width=2)

    # X ticks.
    for p in threads:
        x = x_of(p)
        draw.line((x, top + plot_h, x, top + plot_h + 6), fill="black", width=2)
        draw.text((x, top + plot_h + 12), str(p), fill="black", font=font, anchor="ma")

    draw.text((left + plot_w / 2, height - 32), "Threads p", fill="black", font=bold,
              anchor="mm")
    draw.text((left, top - 30), "Speedup S_p = T1 / Tp", fill="black", font=bold,
              anchor="lm")
    draw.text((width / 2, 38), "Scan / repeats speedup vs threads (measured medians)",
              fill="black", font=bold, anchor="mm")

    # Legend.
    legend_x = left + plot_w - 235
    legend_y = top + 14
    for i, key in enumerate(sorted(series)):
        color, dash, label = STYLE.get(key, ("#333333", None, f"n = {key[0]}, {key[1]}"))
        y = legend_y + i * 32
        if dash:
            dashed_line([(legend_x, y), (legend_x + 34, y)], color, dash, width_px=3)
        else:
            draw.line((legend_x, y, legend_x + 34, y), fill=color, width=3)
        draw.ellipse((legend_x + 12, y - 6, legend_x + 24, y + 6), fill=color)
        draw.text((legend_x + 44, y), label, fill="black", font=font, anchor="lm")

    image.save(path)


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"error: CSV file not found: {CSV_PATH}")
    series = load_rows(CSV_PATH)
    if not series:
        sys.exit(f"error: no parallel rows in {CSV_PATH}")
    try:
        plot_with_matplotlib(series, PNG_PATH)
        backend = "matplotlib"
    except ImportError:
        plot_with_pillow(series, PNG_PATH)
        backend = "Pillow fallback (matplotlib not installed)"
    print(f"wrote {PNG_PATH} using {backend}")


if __name__ == "__main__":
    main()
