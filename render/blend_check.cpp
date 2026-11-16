// Part 3 calculation check: ordered alpha blending on a white pixel.
//
// Build/run from the repository root:
//   ./render/run_blend.sh
//
// Prints the per-channel working for A then B and B then A, then verifies the
// same numbers through the actual render_reference and render_parallel code by
// rendering two circles that both cover the centre pixel of a 100x100 image.
#include "renderer.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct Rgba {
    float r, g, b, a;
};

// Exactly the reference's blend equation, applied to a Pixel.
void blend(Pixel& p, const Rgba& c) {
    p.r = c.a * c.r + (1 - c.a) * p.r;
    p.g = c.a * c.g + (1 - c.a) * p.g;
    p.b = c.a * c.b + (1 - c.a) * p.b;
}

Pixel apply(const std::vector<Rgba>& layers) {
    Pixel p{1, 1, 1};
    for (const Rgba& c : layers) blend(p, c);
    return p;
}

}  // namespace

int main() {
    const Rgba a{1, 0, 0, 0.5f};  // A = (1, 0, 0, 0.5)
    const Rgba b{0, 0, 1, 0.5f};  // B = (0, 0, 1, 0.5)

    const Pixel ab = apply({a, b});
    const Pixel ba = apply({b, a});

    std::printf("white pixel P0 = (1.000000, 1.000000, 1.000000)\n");
    std::printf("A = (r=%.1f, g=%.1f, b=%.1f, alpha=%.1f)\n", a.r, a.g, a.b, a.a);
    std::printf("B = (r=%.1f, g=%.1f, b=%.1f, alpha=%.1f)\n\n", b.r, b.g, b.b, b.a);

    std::printf("A then B:\n");
    std::printf("  after A: R = 0.5*1 + 0.5*1 = %.6f\n", 0.5f * 1 + 0.5f * 1);
    std::printf("           G = 0.5*0 + 0.5*1 = %.6f\n", 0.5f * 0 + 0.5f * 1);
    std::printf("           B = 0.5*0 + 0.5*1 = %.6f\n", 0.5f * 0 + 0.5f * 1);
    std::printf("  after B: R = 0.5*0 + 0.5*1 = %.6f\n", 0.5f * 0 + 0.5f * 1);
    std::printf("           G = 0.5*0 + 0.5*0.5 = %.6f\n", 0.5f * 0 + 0.5f * 0.5f);
    std::printf("           B = 0.5*1 + 0.5*0.5 = %.6f\n", 0.5f * 1 + 0.5f * 0.5f);
    std::printf("  result = (%.6f, %.6f, %.6f)\n\n", ab.r, ab.g, ab.b);

    std::printf("B then A:\n");
    std::printf("  after B: R = 0.5*0 + 0.5*1 = %.6f\n", 0.5f * 0 + 0.5f * 1);
    std::printf("           G = 0.5*0 + 0.5*1 = %.6f\n", 0.5f * 0 + 0.5f * 1);
    std::printf("           B = 0.5*1 + 0.5*1 = %.6f\n", 0.5f * 1 + 0.5f * 1);
    std::printf("  after A: R = 0.5*1 + 0.5*0.5 = %.6f\n", 0.5f * 1 + 0.5f * 0.5f);
    std::printf("           G = 0.5*0 + 0.5*0.5 = %.6f\n", 0.5f * 0 + 0.5f * 0.5f);
    std::printf("           B = 0.5*0 + 0.5*1 = %.6f\n", 0.5f * 0 + 0.5f * 1);
    std::printf("  result = (%.6f, %.6f, %.6f)\n\n", ba.r, ba.g, ba.b);

    const bool same = ab.r == ba.r && ab.g == ba.g && ab.b == ba.b;
    std::printf("orders equal? %s (blending is %s)\n\n", same ? "yes" : "no",
                same ? "commutative" : "NOT commutative");

    // Verify the same numbers through the supplied renderer. Both circles are
    // centred so that they cover the centre pixel of a 100x100 image.
    const int size = 100;
    const Circle ca{0.5f, 0.5f, 0.4f, 1, 0, 0, 0.5f};
    const Circle cb{0.5f, 0.5f, 0.4f, 0, 0, 1, 0.5f};
    const std::vector<Circle> ab_scene{ca, cb};
    const std::vector<Circle> ba_scene{cb, ca};
    const Image ab_image = render_reference(size, size, ab_scene);
    const Image ba_image = render_reference(size, size, ba_scene);
    const Image ab_parallel = render_parallel(size, size, ab_scene);
    const Image ba_parallel = render_parallel(size, size, ba_scene);
    const Pixel& ab_pixel = ab_image.pixels[50 * size + 50];
    const Pixel& ba_pixel = ba_image.pixels[50 * size + 50];
    const Pixel& ab_par_pixel = ab_parallel.pixels[50 * size + 50];
    const Pixel& ba_par_pixel = ba_parallel.pixels[50 * size + 50];

    std::printf("centre pixel (50,50) through render_reference:\n");
    std::printf("  A then B = (%.6f, %.6f, %.6f)\n", ab_pixel.r, ab_pixel.g, ab_pixel.b);
    std::printf("  B then A = (%.6f, %.6f, %.6f)\n", ba_pixel.r, ba_pixel.g, ba_pixel.b);
    std::printf("centre pixel (50,50) through render_parallel:\n");
    std::printf("  A then B = (%.6f, %.6f, %.6f)\n", ab_par_pixel.r, ab_par_pixel.g,
                ab_par_pixel.b);
    std::printf("  B then A = (%.6f, %.6f, %.6f)\n", ba_par_pixel.r, ba_par_pixel.g,
                ba_par_pixel.b);

    const bool matches = std::abs(ab_pixel.r - ab.r) < 1e-6f &&
                         std::abs(ab_pixel.g - ab.g) < 1e-6f &&
                         std::abs(ab_pixel.b - ab.b) < 1e-6f &&
                         std::abs(ba_pixel.r - ba.r) < 1e-6f &&
                         std::abs(ba_pixel.g - ba.g) < 1e-6f &&
                         std::abs(ba_pixel.b - ba.b) < 1e-6f &&
                         std::abs(ab_par_pixel.r - ab_pixel.r) < 1e-6f &&
                         std::abs(ba_par_pixel.r - ba_pixel.r) < 1e-6f;
    std::printf("\nBLEND CHECK: %s\n", matches ? "PASS" : "FAIL");
    return matches ? 0 : 1;
}
