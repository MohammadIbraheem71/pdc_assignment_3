// Part 4 work-model measurements. For each scene this reports the counts that
// explain the timings: circles, pixels, candidate entries (circle-tile pairs),
// candidate tests, actual blends, and the test reduction.
//
// Usage: ./part4_workmodel [size] [tile] [scene[,scene...]]
// Defaults: 1024 16 (all Part 4 scenes)
//
// Tests: every pixel in a tile calls covers() once per candidate circle, so
// candidate_tests = width*height*c_avg exactly. Blends are counted once per
// circle/pixel pair with the same float covers() test. The renderer itself is
// only used for its candidate statistics; the covered-pixel count is
// independent of circle order, so summing per-circle coverage is exact.
#include "part4_scenes.h"
#include "renderer.h"

#include <omp.h>

#include <algorithm>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Same float arithmetic as the reference covers(), counted once per circle.
long long covered_pixels(const Circle& c, int width, int height) {
    if (width <= 0 || height <= 0) return 0;
    const float r = std::fabs(c.r);
    const double radius_px = static_cast<double>(r) * width;
    const int cx = static_cast<int>(std::floor(static_cast<double>(c.x) * width));
    const int cy = static_cast<int>(std::floor(static_cast<double>(c.y) * height));
    const int pad = static_cast<int>(std::ceil(radius_px)) + 2;
    const int x0 = std::max(0, cx - pad), x1 = std::min(width - 1, cx + pad);
    const int y0 = std::max(0, cy - pad), y1 = std::min(height - 1, cy + pad);
    long long count = 0;
    for (int y = y0; y <= y1; ++y) {
        const float py = (y + 0.5f) / height;
        const float dy = py - c.y;
        for (int x = x0; x <= x1; ++x) {
            const float px = (x + 0.5f) / width;
            const float dx = px - c.x;
            if (dx * dx + dy * dy <= c.r * c.r) ++count;
        }
    }
    return count;
}

}  // namespace

int main(int argc, char** argv) {
    const int size = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int tile = argc > 2 ? std::atoi(argv[2]) : 16;
    std::vector<std::string> scenes;
    if (argc > 3) {
        std::stringstream in(argv[3]);
        std::string item;
        while (std::getline(in, item, ','))
            if (!item.empty()) scenes.push_back(item);
    } else {
        scenes = part4_scene_names();
    }
    if (size < 1 || tile < 1 || scenes.empty()) {
        std::fprintf(stderr, "usage: ./part4_workmodel [size >=1] [tile >=1] "
                             "[scene[,scene...]]\n");
        return 2;
    }

    std::printf("# Part 4 work model, %dx%d, tile %d\n", size, size, tile);
    std::printf("# baseline tests = width*height*Nc; candidate tests = width*height*c_avg\n");
    std::printf("# blends = circle/pixel pairs that actually pass covers()\n");
    std::printf("%-10s %5s %10s %10s %9s %10s %12s %12s %8s %9s\n", "scene", "Nc", "tiles",
                "entries", "c_avg", "t_avg", "base_tests", "cand_tests", "test_red",
                "blends");

    for (const std::string& scene : scenes) {
        std::vector<Circle> circles;
        if (!make_any_scene(scene, circles)) {
            std::fprintf(stderr, "unknown scene '%s'\n", scene.c_str());
            return 2;
        }
        RenderStats stats;
        render_parallel_optimized(size, size, circles, tile, RenderSchedule::Static, 0,
                                  &stats);
        const double pixels = static_cast<double>(size) * size;
        const double c_avg = stats.tile_count > 0
                                 ? static_cast<double>(stats.candidate_entries) /
                                       stats.tile_count
                                 : 0.0;
        const double t_avg = circles.empty()
                                 ? 0.0
                                 : static_cast<double>(stats.candidate_entries) /
                                       circles.size();
        const double base_tests = pixels * circles.size();
        const double cand_tests = pixels * c_avg;
        long long blends = 0;
        #pragma omp parallel for schedule(static) reduction(+:blends)
        for (int i = 0; i < static_cast<int>(circles.size()); ++i)
            blends += covered_pixels(circles[static_cast<std::size_t>(i)], size, size);
        std::printf("%-10s %5zu %10lld %10lld %9.3f %10.3f %12.0f %12.0f %7.1fx %9lld\n",
                    scene.c_str(), circles.size(), stats.tile_count,
                    stats.candidate_entries, c_avg, t_avg, base_tests, cand_tests,
                    cand_tests > 0 ? base_tests / cand_tests : 0.0, blends);
    }
    return 0;
}
