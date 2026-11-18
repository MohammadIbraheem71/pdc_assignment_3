// Part 4 correctness and determinism tests. Separate from the starter tests.
//
// Build/run from the repository root:
//   ./render/run_part4_tests.sh
//
// For every Part 4 and starter scene this checks the baseline, the optimized
// renderer and the reference at 1, 2, 4 and 8 threads. It also sweeps tile sizes
// (including sizes that do not divide the image), compares static/dynamic/guided
// schedules, and covers zero/one-pixel images, zero circles, circles fully or
// partly outside the image, circles straddling tile borders, circles exactly on
// a tile boundary, zero/negative radius, and many concentric circles.
//
// Every optimized result must pass images_close, be bit-exact with the reference
// (same per-pixel order and float operations), and be identical over repeated
// renders. Output is saved to part4_tests.log by the run script.
#include "part4_scenes.h"
#include "renderer.h"
#include "../common/rng.h"

#include <omp.h>

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

void report(const std::string& label, int width, int height, int tile, int threads,
            const char* schedule, bool close, bool exact, bool deterministic) {
    ++checks;
    const bool ok = close && exact && deterministic;
    if (!ok) ++failures;
    std::printf("%-34s %4dx%-4d tile=%-3d p=%-2d %-10s close=%-4s exact=%-4s det=%-4s %s\n",
                label.c_str(), width, height, tile, threads, schedule,
                close ? "PASS" : "FAIL", exact ? "PASS" : "FAIL",
                deterministic ? "PASS" : "FAIL", ok ? "PASS" : "FAIL");
}

// Checks the baseline once at `threads` and the optimized renderer at every
// requested thread count. The caller supplies the reference image.
bool check_optimized(const std::string& label, int width, int height, int tile,
                     RenderSchedule schedule, int chunk, const char* schedule_name,
                     const std::vector<Circle>& circles, const std::vector<int>& threads,
                     const Image& reference) {
    bool all_ok = true;
    for (int p : threads) {
        omp_set_num_threads(p);
        RenderStats stats;
        const Image first = render_parallel_optimized(width, height, circles, tile, schedule,
                                                      chunk, &stats);
        bool deterministic = true;
        for (int repeat = 0; repeat < 3; ++repeat) {
            const Image again = render_parallel_optimized(width, height, circles, tile,
                                                          schedule, chunk, nullptr);
            if (!exactly_equal(first, again)) deterministic = false;
        }
        const bool close = images_close(reference, first);
        const bool exact = exactly_equal(reference, first);
        const bool stats_ok = stats.preprocess_ms >= 0.0 && stats.render_ms >= 0.0 &&
                              stats.tile_count >= 0 && stats.min_candidates <= stats.max_candidates;
        report(label, width, height, tile, p, schedule_name, close, exact,
               deterministic && stats_ok);
        all_ok = all_ok && close && exact && deterministic && stats_ok;
    }
    return all_ok;
}

void check_case(const std::string& label, int width, int height, int tile,
                const std::vector<Circle>& circles, const std::vector<int>& threads) {
    const Image reference = render_reference(width, height, circles);
    const int baseline_threads = threads.size() > 1 ? threads[1] : threads[0];

    omp_set_num_threads(baseline_threads);
    const Image baseline = render_parallel(width, height, circles);
    report(label + " baseline", width, height, tile, baseline_threads, "runtime",
           images_close(reference, baseline), exactly_equal(reference, baseline), true);

    check_optimized(label, width, height, tile, RenderSchedule::Static, 0, "static",
                    circles, threads, reference);
}

}  // namespace

int main() {
    const int available = omp_get_num_procs();
    omp_set_dynamic(0);
    std::vector<int> threads;
    for (int p : {1, 2, 4, 8}) {
        if (p <= available)
            threads.push_back(p);
        else
            std::printf("# note: %d threads skipped, only %d processors available\n", p,
                        available);
    }

    std::printf("RENDER Part 4 correctness/determinism tests\n");
    std::printf("processors available: %d, thread counts tested: ", available);
    for (std::size_t i = 0; i < threads.size(); ++i)
        std::printf("%s%d", i == 0 ? "" : ",", threads[i]);
    std::printf("\n\n");

    // 1. All Part 4 and starter scenes on a non-square image whose size is not
    // divisible by the default tile size.
    for (const std::string& scene : all_scene_names()) {
        std::vector<Circle> circles;
        if (!make_any_scene(scene, circles)) {
            std::printf("# internal error: scene %s not found\n", scene.c_str());
            ++failures;
            continue;
        }
        const int width = 257, height = 193, tile = 16;
        const Image reference = render_reference(width, height, circles);

        omp_set_num_threads(4);
        const Image baseline = render_parallel(width, height, circles);
        report(scene + " baseline", width, height, tile, 4, "runtime",
               images_close(reference, baseline), exactly_equal(reference, baseline), true);

        check_optimized(scene, width, height, tile, RenderSchedule::Static, 0, "static",
                        circles, threads, reference);
    }

    std::printf("\n-- tile-size sweep on non-divisible images --\n");
    for (const char* scene_text : {"p4_corner", "p4_heavy"}) {
        const std::string scene = scene_text;
        std::vector<Circle> circles;
        make_any_scene(scene, circles);
        const int width = 257, height = 193;
        const Image reference = render_reference(width, height, circles);
        for (int tile : {1, 2, 3, 7, 8, 16, 32, 64, 128, 512})
            check_optimized(scene + " tile", width, height, tile, RenderSchedule::Static, 0,
                            "static", circles, threads, reference);
    }

    std::printf("\n-- schedule comparison on p4_corner --\n");
    {
        std::vector<Circle> circles;
        make_any_scene("p4_corner", circles);
        const int width = 257, height = 193;
        const Image reference = render_reference(width, height, circles);
        check_optimized("p4_corner dynamic", width, height, 16, RenderSchedule::Dynamic, 1,
                        "dynamic,1", circles, threads, reference);
        check_optimized("p4_corner dynamic", width, height, 16, RenderSchedule::Dynamic, 4,
                        "dynamic,4", circles, threads, reference);
        check_optimized("p4_corner guided", width, height, 16, RenderSchedule::Guided, 4,
                        "guided,4", circles, threads, reference);
    }

    std::printf("\n-- edge cases --\n");
    const std::vector<Circle> empty;
    check_case("zero circles 33x17", 33, 17, 16, empty, threads);
    check_case("zero circles 0x0", 0, 0, 16, empty, threads);
    check_case("zero circles 0x7", 0, 7, 16, empty, threads);
    check_case("zero circles 7x0", 7, 0, 16, empty, threads);
    check_case("one pixel covered", 1, 1, 16,
               {Circle{0.5f, 0.5f, 0.2f, 1, 0, 0, 0.5f}}, threads);

    const std::vector<Circle> outside{
        Circle{-0.5f, -0.5f, 0.1f, 1, 0, 0, 0.5f},
        Circle{1.5f, 0.5f, 0.2f, 0, 1, 0, 0.5f},
        Circle{0.5f, 1.6f, 0.1f, 0, 0, 1, 0.5f},
        Circle{-2.0f, 2.0f, 0.3f, 1, 1, 0, 0.4f}};
    check_case("fully outside", 80, 80, 16, outside, threads);

    const std::vector<Circle> partly{
        Circle{-0.05f, 0.5f, 0.2f, 1, 0, 0, 0.5f},
        Circle{0.5f, 1.02f, 0.25f, 0, 0, 1, 0.5f},
        Circle{1.05f, 1.05f, 0.3f, 0, 1, 0, 0.5f}};
    check_case("partly outside", 80, 80, 16, partly, threads);

    // A circle whose centre is exactly on the boundary between two tiles, and a
    // circle whose covered pixel centre lies exactly on a tile boundary (pixel 48
    // is the first pixel of the fourth 16-pixel tile). Radius 0.25 = 16 pixels,
    // and the boundary pixel is exactly at distance 16/64 from the centre.
    const std::vector<Circle> straddle{
        Circle{0.25f, 0.25f, 0.03f, 1, 0, 0, 0.5f},
        Circle{32.5f / 64, 32.5f / 64, 0.25f, 0, 0, 1, 0.5f},
        Circle{0.75f, 0.75f, 0.02f, 0, 1, 0, 0.5f}};
    check_case("tile boundary exact", 64, 64, 16, straddle, threads);

    check_case("zero radius", 64, 64, 16,
               {Circle{32.5f / 64, 32.5f / 64, 0.0f, 1, 0, 0, 0.7f}}, threads);
    check_case("negative radius", 64, 64, 16,
               {Circle{0.5f, 0.5f, -0.2f, 0, 1, 0, 0.5f}}, threads);

    // Many concentric circles with different colors/opacities: catches any
    // accidental reordering or circle-parallel blending.
    {
        std::vector<Circle> concentric;
        Lcg rng(987654);
        for (int i = 0; i < 64; ++i)
            concentric.push_back(Circle{0.5f, 0.5f, 0.4f, rng.next_float(),
                                        rng.next_float(), rng.next_float(),
                                        rng.range(0.1f, 0.9f)});
        check_case("64 concentric circles", 96, 96, 16, concentric, threads);
        check_case("64 concentric tile 7", 96, 96, 7, concentric, threads);
    }

    std::printf("\nRENDER PART 4 TESTS: %d checks, %d failed -> %s\n", checks, failures,
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
