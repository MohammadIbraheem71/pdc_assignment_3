// Part 3 extra tests. Separate from the supplied demo and benchmark.
//
// Build/run from the repository root:
//   ./render/run_tests.sh
//
// Covers overlapping translucent circles, mostly-empty scenes, a single circle,
// zero circles, circles partly or fully outside the image, many circles at the
// same position with different colors (ordering stress), non-square images, and
// deterministic re-rendering at 1, 2, 4 and 8 threads. Every case is compared
// with render_reference using images_close (the supplied tolerance) and also
// checked for bit-exact equality, because the parallel version performs the same
// float operations in the same per-pixel order.
#include "renderer.h"
#include "../common/rng.h"

#include <omp.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int checks = 0;
int failures = 0;

bool exactly_equal(const Image& a, const Image& b) {
    if (a.width != b.width || a.height != b.height || a.pixels.size() != b.pixels.size())
        return false;
    for (std::size_t i = 0; i < a.pixels.size(); ++i)
        if (a.pixels[i].r != b.pixels[i].r || a.pixels[i].g != b.pixels[i].g ||
            a.pixels[i].b != b.pixels[i].b)
            return false;
    return true;
}

void check_case(const std::string& name, int width, int height,
                const std::vector<Circle>& circles, const std::vector<int>& threads) {
    const Image reference = render_reference(width, height, circles);

    for (int p : threads) {
        omp_set_num_threads(p);
        const Image first = render_parallel(width, height, circles);

        // Re-render several times to confirm determinism.
        bool deterministic = true;
        for (int repeat = 0; repeat < 4; ++repeat) {
            const Image again = render_parallel(width, height, circles);
            if (!exactly_equal(first, again)) deterministic = false;
        }

        const bool close = images_close(reference, first);
        const bool exact = exactly_equal(reference, first);
        const bool ok = close && deterministic;
        ++checks;
        if (!ok) ++failures;
        std::printf("%-32s %4dx%-4d circles=%-4zu p=%-2d close=%-4s exact=%-4s "
                    "deterministic=%-4s %s\n",
                    name.c_str(), width, height, circles.size(), p, close ? "PASS" : "FAIL",
                    exact ? "PASS" : "FAIL", deterministic ? "PASS" : "FAIL",
                    ok ? "PASS" : "FAIL");
    }
}

}  // namespace

int main() {
    const int available = omp_get_num_procs();
    std::vector<int> threads;
    for (int p : {1, 2, 4, 8}) {
        if (p <= available)
            threads.push_back(p);
        else
            std::printf("# note: %d threads skipped, only %d processors available\n", p,
                        available);
    }

    std::printf("RENDER Part 3 edge-case tests\n");
    std::printf("processors available: %d, thread counts tested: ", available);
    for (std::size_t i = 0; i < threads.size(); ++i)
        std::printf("%s%d", i == 0 ? "" : ",", threads[i]);
    std::printf("\n\n");

    // Zero circles: the image must stay white, including non-square sizes.
    check_case("zero circles square", 64, 64, {}, threads);
    check_case("zero circles non-square", 37, 21, {}, threads);

    // Degenerate sizes.
    check_case("zero width and height", 0, 0, {}, threads);
    check_case("zero width only", 0, 7, {}, threads);

    // One circle.
    check_case("single circle centre", 129, 129,
               {Circle{0.5f, 0.5f, 0.25f, 0.3f, 0.6f, 0.9f, 0.7f}}, threads);
    check_case("single circle corner", 64, 64,
               {Circle{0.0f, 0.0f, 0.2f, 1.0f, 0.0f, 0.0f, 0.5f}}, threads);
    check_case("single circle edge", 64, 64,
               {Circle{0.5f, 0.0f, 0.3f, 0.0f, 1.0f, 0.0f, 0.4f}}, threads);

    // Most pixels empty: a few tiny circles.
    check_case("mostly empty", 257, 193,
               {Circle{0.1f, 0.2f, 0.01f, 1.0f, 0.0f, 0.0f, 0.8f},
                Circle{0.7f, 0.6f, 0.008f, 0.0f, 1.0f, 0.0f, 0.8f},
                Circle{0.9f, 0.1f, 0.012f, 0.0f, 0.0f, 1.0f, 0.8f}},
               threads);

    // Overlapping translucent circles: A then B and B then A.
    check_case("overlap A then B", 64, 64,
               {Circle{0.45f, 0.5f, 0.35f, 1.0f, 0.0f, 0.0f, 0.5f},
                Circle{0.55f, 0.5f, 0.35f, 0.0f, 0.0f, 1.0f, 0.5f}},
               threads);
    check_case("overlap B then A", 64, 64,
               {Circle{0.55f, 0.5f, 0.35f, 0.0f, 0.0f, 1.0f, 0.5f},
                Circle{0.45f, 0.5f, 0.35f, 1.0f, 0.0f, 0.0f, 0.5f}},
               threads);

    // Circles partly or fully outside the image.
    check_case("partly outside", 80, 80,
               {Circle{-0.05f, 0.5f, 0.2f, 1.0f, 0.0f, 0.0f, 0.5f},
                Circle{0.5f, 1.02f, 0.25f, 0.0f, 0.0f, 1.0f, 0.5f},
                Circle{1.05f, 1.05f, 0.3f, 0.0f, 1.0f, 0.0f, 0.5f}},
               threads);
    check_case("fully outside", 80, 80,
               {Circle{-0.5f, -0.5f, 0.1f, 1.0f, 0.0f, 0.0f, 0.5f},
                Circle{1.5f, 0.5f, 0.1f, 0.0f, 1.0f, 0.0f, 0.5f},
                Circle{0.5f, 1.6f, 0.1f, 0.0f, 0.0f, 1.0f, 0.5f}},
               threads);

    // Ordering stress: many concentric circles with different colors and
    // opacities. A different order gives a different image, so this catches any
    // accidental reordering, sorting, or circle-parallel blending.
    {
        std::vector<Circle> same_position;
        Lcg rng(987654);
        for (int i = 0; i < 64; ++i)
            same_position.push_back(Circle{0.5f, 0.5f, 0.4f, rng.next_float(),
                                           rng.next_float(), rng.next_float(),
                                           rng.range(0.1f, 0.9f)});
        check_case("same position 64 circles", 96, 96, same_position, threads);
    }

    // A larger mixed scene with deterministic random circles, including some
    // that overlap heavily and some outside the frame.
    {
        std::vector<Circle> mixed;
        Lcg rng(13579);
        for (int i = 0; i < 120; ++i) {
            const float x = rng.range(-0.1f, 1.1f);
            const float y = rng.range(-0.1f, 1.1f);
            const float r = rng.range(0.02f, 0.2f);
            mixed.push_back(Circle{x, y, r, rng.next_float(), rng.next_float(),
                                   rng.next_float(), rng.range(0.05f, 0.95f)});
        }
        check_case("mixed random 120", 200, 150, mixed, threads);
    }

    std::printf("\nRENDER EDGE TESTS: %d checks, %d failed -> %s\n", checks, failures,
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
