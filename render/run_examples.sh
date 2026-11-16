#!/usr/bin/env bash
# Renders two fixed-seed scenes at 512x512 and keeps the reference/parallel
# images in part3_images/ for the report. Deterministic: the scenes come from
# the fixed seeds in render/scenes.cpp and no CLI input is needed.
#
# Usage: ./render/run_examples.sh
# Output: part3_images/{dense,clustered}_512_{reference,parallel}.pnm and .png
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/part3_images"
mkdir -p "$out"

cd "$root"
for scene in dense clustered; do
    ./render_demo 512 "$scene" parallel
    mv "render_${scene}_reference.ppm" "$out/${scene}_512_reference.pnm"
    mv "render_${scene}_parallel.ppm" "$out/${scene}_512_parallel.pnm"
done

# The starter writes text PPM (P3). Convert the copies to PNG when Pillow is
# available so they can be pasted into the report directly.
python3 - "$out" <<'PY'
import os
import sys

out = sys.argv[1]
try:
    from PIL import Image
except ImportError:
    print("Pillow not installed: kept only the .pnm files")
    sys.exit(0)

for name in sorted(os.listdir(out)):
    if not name.endswith(".pnm"):
        continue
    path = os.path.join(out, name)
    with open(path) as handle:
        tokens = handle.read().split()
    if tokens[0] != "P3":
        raise SystemExit(f"unsupported PPM format in {path}")
    width, height, maximum = map(int, tokens[1:4])
    values = list(map(int, tokens[4:]))
    if len(values) != width * height * 3:
        raise SystemExit(f"bad pixel count in {path}")
    if maximum != 255:
        values = [round(v * 255 / maximum) for v in values]
    image = Image.frombytes("RGB", (width, height), bytes(values))
    image.save(path[:-4] + ".png")
    print(f"wrote {path[:-4]}.png")
    os.remove(path)  # the PNG is a lossless copy of the 8-bit PPM values
PY
