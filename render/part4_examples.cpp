// Part 4 example renderer for report images. Writes reference, baseline and
// optimized PPM files for one scene. The shell script converts them to PNG.
//
// Usage: ./part4_examples [size] [scene] [output-directory]
#include "part4_scenes.h"
#include "renderer.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const int size = argc > 1 ? std::atoi(argv[1]) : 512;
    const std::string scene = argc > 2 ? argv[2] : "p4_corner";
    const std::string outdir = argc > 3 ? argv[3] : "part4_images";
    if (size < 1 || size > 4096) {
        std::fprintf(stderr, "usage: ./part4_examples [size 1..4096] [scene] [outdir]\n");
        return 2;
    }
    std::vector<Circle> circles;
    if (!make_any_scene(scene, circles)) {
        std::fprintf(stderr, "unknown scene '%s'\n", scene.c_str());
        return 2;
    }

    const Image reference = render_reference(size, size, circles);
    const Image baseline = render_parallel(size, size, circles);
    const Image optimized = render_parallel_optimized(size, size, circles, 16,
                                                      RenderSchedule::Static, 0, nullptr);
    const bool baseline_ok = images_close(reference, baseline);
    const bool optimized_ok = images_close(reference, optimized);

    const std::string prefix = outdir + "/" + scene + "_" + std::to_string(size);
    save_ppm(reference, (prefix + "_reference.ppm").c_str());
    save_ppm(baseline, (prefix + "_baseline.ppm").c_str());
    save_ppm(optimized, (prefix + "_optimized.ppm").c_str());
    std::printf("%s %dx%d circles=%zu baseline=%s optimized=%s\n", scene.c_str(), size, size,
                circles.size(), baseline_ok ? "PASS" : "FAIL",
                optimized_ok ? "PASS" : "FAIL");
    return baseline_ok && optimized_ok ? 0 : 1;
}
