#!/usr/bin/env bash
# Renders deterministic Part 4 example images at 512x512 into part4_images/.
# Files: <scene>_512_{reference,baseline,optimized}.png
# The PNGs are lossless conversions of the starter's text PPM output.
#
# Usage: ./render/run_part4_examples.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/render/.part4_build"
out="$root/part4_images"
mkdir -p "$build" "$out"

cd "$root"
g++ -O2 -std=c++17 -Wall -Wextra -fopenmp \
    render/part4_examples.cpp render/part4_scenes.cpp render/scenes.cpp render/renderer.cpp \
    -o "$build/part4_examples"

for scene in p4_sparse p4_heavy p4_corner; do
    "$build/part4_examples" 512 "$scene" "$out"
done

python3 - "$out" <<'PY'
import os
import sys

out = sys.argv[1]
try:
    from PIL import Image
except ImportError:
    print("Pillow not installed: kept only the .ppm files")
    sys.exit(0)

for name in sorted(os.listdir(out)):
    if not name.endswith(".ppm"):
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
    os.remove(path)
PY

echo "example images are in $out (delete before zipping the submission)"
