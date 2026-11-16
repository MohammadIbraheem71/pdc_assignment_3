#!/usr/bin/env python3
"""Plot SAXPY speedup vs threads from the benchmark CSV.

Usage:
    python3 saxpy/plot_speedup.py [saxpy_results.csv] [saxpy_speedup.png]

matplotlib is used when it is installed; otherwise the script falls back to
Pillow (PIL) so the PNG can still be produced from the measured CSV.
"""
import csv
import math
import os
import sys

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "saxpy_results.csv"
PNG_PATH = sys.argv[2] if len(sys.argv) > 2 else "saxpy_speedup.png"

STYLE = {
    10000: ("#1f77b4", "n = 1e4"),
    1000000: ("#d95f02", "n = 1e6"),
    10000000: ("#2ca02c", "n = 1e7"),
}


def load_rows(path):
    series = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            n = int(row["n"])
            series.setdefault(n, []).append(
                (int(row["threads"]), float(row["speedup"]))
            )
    for points in series.values():
        points.sort()
    return series


def nice_ceiling(value, step=0.2):
    return math.ceil(value / step) * step


def plot_with_matplotlib(series, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for n in sorted(series):
        color, label = STYLE.get(n, (None, f"n = {n}"))
        xs = [p for p, _ in series[n]]
        ys = [s for _, s in series[n]]
        ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=label)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="serial (S = 1)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlim(0.9, 8.8)
    ax.set_ylim(0, nice_ceiling(max(s for pts in series.values() for _, s in pts) + 0.2))
    ax.set_xlabel("Threads p")
    ax.set_ylabel("Speedup $S_p = T_1 / T_p$")
    ax.set_title("SAXPY speedup vs threads (measured medians)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_with_pillow(series, path):
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1200, 800
    left, right, top, bottom = 110, 60, 90, 100
    plot_w = width - left - right
    plot_h = height - top - bottom

    threads = sorted({p for pts in series.values() for p, _ in pts})
    max_speedup = max(s for pts in series.values() for _, s in pts)
    y_max = nice_ceiling(max(max_speedup + 0.2, 1.2))

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
    steps = 5
    for i in range(steps + 1):
        value = y_max * i / steps
        y = y_of(value)
        draw.line((left, y, left + plot_w, y), fill=(220, 220, 220), width=1)
        draw.text((left - 12, y), f"{value:.1f}", fill="black", font=font, anchor="rm")

    # Axes.
    draw.line((left, top, left, top + plot_h), fill="black", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=2)

    # Serial baseline.
    y_one = y_of(1.0)
    for x in range(left, left + plot_w, 14):
        draw.line((x, y_one, min(x + 7, left + plot_w), y_one), fill=(120, 120, 120), width=2)
    draw.text((left + plot_w - 8, y_of(1.05)), "serial (S = 1)", fill=(90, 90, 90),
              font=font, anchor="rb")

    for n in sorted(series):
        color, _ = STYLE.get(n, ("#333333", ""))
        points = [(x_of(p), y_of(s)) for p, s in series[n]]
        if len(points) > 1:
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
    draw.text((width / 2, 38), "SAXPY speedup vs threads (measured medians)", fill="black",
              font=bold, anchor="mm")

    # Legend.
    legend_x = left + plot_w - 190
    legend_y = top + 16
    for i, n in enumerate(sorted(series)):
        color, label = STYLE.get(n, ("#333333", f"n = {n}"))
        y = legend_y + i * 34
        draw.line((legend_x, y, legend_x + 34, y), fill=color, width=3)
        draw.ellipse((legend_x + 12, y - 6, legend_x + 24, y + 6), fill=color)
        draw.text((legend_x + 44, y), label, fill="black", font=font, anchor="lm")

    image.save(path)


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"error: CSV file not found: {CSV_PATH}")
    series = load_rows(CSV_PATH)
    if not series:
        sys.exit(f"error: no rows in {CSV_PATH}")
    try:
        plot_with_matplotlib(series, PNG_PATH)
        backend = "matplotlib"
    except ImportError:
        plot_with_pillow(series, PNG_PATH)
        backend = "Pillow fallback (matplotlib not installed)"
    print(f"wrote {PNG_PATH} using {backend}")


if __name__ == "__main__":
    main()
