# Part 3: Correct Parallel Circle Rendering

Part 3 implements `render_parallel()` in `render/renderer.cpp`. It is the simple,
correct baseline that Part 4 will optimize and compare against. Nothing outside
`render/` was changed for the implementation, and the serial reference, the supplied
demo/benchmark, the image comparison, and all signatures are untouched.

The implementation decomposes the image by rows:

```cpp
#pragma omp parallel for schedule(runtime)
for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
        Pixel p{1, 1, 1};
        for (const Circle& c : circles)          // original order, 0..N-1
            if (covers(c, x, y, width, height))  // exact reference test
                blend(p, c);                     // exact reference equation
        image.pixels[y * width + x] = p;         // one write per pixel
    }
}
```

Each pixel is owned by exactly one thread, circles are never reordered, and the pixel
color is accumulated in a local variable so no atomics or locks are needed.
`schedule(runtime)` is used so `OMP_SCHEDULE` can change the schedule in Part 4.

## Requirements

Same as the starter: C++17 with OpenMP (`g++` recommended). The examples below assume
you are in the repository root.

## Compile

```sh
make render_demo
```

To build everything, use `make`. If you override `CXXFLAGS`, keep `-fopenmp`.

## Run the supplied demo

```sh
./render_demo 256 dense parallel
./render_demo 256 sparse parallel
./render_demo 256 overlap parallel
./render_demo 256 clustered parallel

# Same scenes at other thread counts (the renderer never sets the thread count):
OMP_NUM_THREADS=1 ./render_demo 512 clustered parallel
OMP_NUM_THREADS=4 ./render_demo 512 clustered parallel
OMP_NUM_THREADS=8 ./render_demo 512 clustered parallel
```

The demo prints `serial_ms`, `parallel_ms`, and `correctness=PASS` (exit 0) or
`correctness=FAIL` (exit 1), and writes `render_<scene>_reference.ppm` and
`render_<scene>_parallel.ppm` to the current directory.

Usage: `./render_demo [size] [scene] [variant]`, where `scene` is one of
`dense`, `sparse`, `overlap`, `clustered` and `variant` is `parallel` (Part 3) or
`optimized` (Part 4, still a stub).

## Extra edge-case and determinism tests

`render/edge_tests.cpp` is a separate program; it does not modify the supplied tests or
comparison code. It covers zero circles, empty and non-square images, single circles,
mostly-empty scenes, overlapping circles in both orders, circles partly/fully outside the
image, 64 concentric circles with different colors/opacities, and a mixed random scene.
At every thread count it checks `images_close`, bit-exact equality with the reference,
and repeated renders for determinism.

```sh
./render/run_tests.sh
```

Output is saved to `part3_tests.log`. Expected last line:
`RENDER EDGE TESTS: 56 checks, 0 failed -> PASS`.

## Ordered-blending calculation check

```sh
./render/run_blend.sh
```

Prints the per-channel working for A then B and B then A, verifies that the same values
come out of `render_reference` and `render_parallel`, and saves `part3_blend.txt`.
Expected result:

- A then B: `(0.500000, 0.250000, 0.750000)`
- B then A: `(0.750000, 0.250000, 0.500000)`

## Light timing sanity check

Full benchmarking belongs to Part 4. This check uses one scene, one warm-up plus 7 timed
runs, medians, `omp_get_wtime()`, and no I/O inside the timed region.

```sh
./render/run_timing.sh              # 512x512 clustered, 7 runs
./render/run_timing.sh 256 5 sparse # custom size, runs (>=5), scene
```

Output is saved to `part3_timing.txt`. It reports the serial reference and
`render_parallel` at 1, 2, 4 and 8 threads, with `Sp = T1/Tp` and `vs_serial`.

## Per-thread load probe

```sh
./render/run_load.sh                # 1024x1024 clustered at 8 then 4 threads
./render/run_load.sh 1024 clustered 8
```

Replicates the static contiguous row split, counts covered pixels and blends per thread,
and times each thread's own loop after a barrier. Output is saved to `part3_load.txt`.
This is a diagnostic, not part of the graded renderer.

## Reproducible example images

The scenes come from fixed seeds in `render/scenes.cpp`, so these commands always render
the same images:

```sh
./render/run_examples.sh
```

This renders `dense` and `clustered` at 512x512 and writes reference/parallel PNGs into
`part3_images/`:

- `dense_512_reference.png`, `dense_512_parallel.png`
- `clustered_512_reference.png`, `clustered_512_parallel.png`

The PNGs are lossless conversions of the starter's text PPM output (requires Pillow).
The reference and parallel files for a scene are byte-identical. Copy only the images you
need into the report, and delete the `part3_images/` folder before zipping the
submission.

## Analysis

`part3_notes.md` contains the ordered-blending working, the lost-update race interleaving
table, the circle-parallel vs pixel-parallel discussion, the 1024x1024 load calculation
with measured per-thread numbers, and a clear separation of measured conclusions from
reasoning/assumptions.

## Clean up

```sh
make clean
rm -rf render/.part3_build
```

`make clean` removes the six binaries and `render_*.ppm` in the repository root. The
extra Part 3 programs are built by the scripts into `render/.part3_build`, which
`make clean` does not know about, so remove that directory manually. Keep
`part3_images/`, `part3_tests.log`, `part3_timing.txt`, `part3_load.txt`,
`part3_blend.txt`, `part3_notes.md`, and this file for the report; delete the images
before zipping.

## Part 4 note

`render_parallel()` is the frozen Part 3 baseline (also tagged `part3-baseline` in git).
Part 4 should implement `render_optimized()` only, leaving `render_parallel()` unchanged
so the benchmark can compare the baseline with the optimized renderer.
