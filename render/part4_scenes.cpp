// Part 4 deterministic scene generators (fixed Lcg seeds).
#include "part4_scenes.h"
#include "scenes.h"
#include "../common/rng.h"

namespace {

Circle random_circle(Lcg& rng, float x, float y, float r, float alpha_lo, float alpha_hi) {
    return Circle{x, y, r, rng.next_float(), rng.next_float(), rng.next_float(),
                  rng.range(alpha_lo, alpha_hi)};
}

float bell_offset(Lcg& rng, float spread) {
    float sum = rng.next_float();
    sum += rng.next_float();
    sum += rng.next_float();
    return (sum - 1.5f) / 1.5f * spread;
}

void make_sparse(std::vector<Circle>& circles) {
    Lcg rng(41001);
    for (int i = 0; i < 2000; ++i) {
        const float x = rng.next_float();
        const float y = rng.next_float();
        const float r = rng.range(0.002f, 0.006f);
        circles.push_back(random_circle(rng, x, y, r, 0.4f, 0.9f));
    }
}

void make_heavy(std::vector<Circle>& circles) {
    Lcg rng(42002);
    for (int i = 0; i < 400; ++i) {
        const float x = rng.range(0.08f, 0.92f);
        const float y = rng.range(0.08f, 0.92f);
        const float r = rng.range(0.18f, 0.40f);
        circles.push_back(random_circle(rng, x, y, r, 0.05f, 0.18f));
    }
}

void make_corner(std::vector<Circle>& circles) {
    Lcg rng(43003);
    // Most circles are packed into the top-left corner; a few are scattered so
    // every part of the image has some work.
    for (int i = 0; i < 700; ++i) {
        const float x = 0.18f + bell_offset(rng, 0.10f);
        const float y = 0.18f + bell_offset(rng, 0.10f);
        const float r = rng.range(0.006f, 0.018f);
        circles.push_back(random_circle(rng, x, y, r, 0.2f, 0.6f));
    }
    for (int i = 0; i < 60; ++i) {
        const float x = rng.next_float();
        const float y = rng.next_float();
        const float r = rng.range(0.005f, 0.015f);
        circles.push_back(random_circle(rng, x, y, r, 0.2f, 0.6f));
    }
}

void make_few(std::vector<Circle>& circles) {
    Lcg rng(44004);
    for (int i = 0; i < 8; ++i) {
        const float x = rng.range(0.2f, 0.8f);
        const float y = rng.range(0.2f, 0.8f);
        const float r = rng.range(0.10f, 0.25f);
        circles.push_back(random_circle(rng, x, y, r, 0.2f, 0.6f));
    }
}

void make_one(std::vector<Circle>& circles) {
    Lcg rng(45005);
    circles.push_back(random_circle(rng, 0.5f, 0.5f, 0.7f, 0.4f, 0.6f));
}

void make_huge(std::vector<Circle>& circles) {
    Lcg rng(46006);
    for (int i = 0; i < 32; ++i) {
        const float x = 0.5f + bell_offset(rng, 0.08f);
        const float y = 0.5f + bell_offset(rng, 0.08f);
        const float r = rng.range(0.55f, 0.90f);
        circles.push_back(random_circle(rng, x, y, r, 0.03f, 0.10f));
    }
}

}  // namespace

std::vector<std::string> part4_scene_names() {
    return {"p4_sparse", "p4_heavy", "p4_corner", "p4_few", "p4_one", "p4_huge"};
}

bool make_part4_scene(const std::string& name, std::vector<Circle>& circles) {
    std::vector<Circle> built;
    if (name == "p4_sparse") make_sparse(built);
    else if (name == "p4_heavy") make_heavy(built);
    else if (name == "p4_corner") make_corner(built);
    else if (name == "p4_few") make_few(built);
    else if (name == "p4_one") make_one(built);
    else if (name == "p4_huge") make_huge(built);
    else return false;
    circles = built;
    return true;
}

bool make_any_scene(const std::string& name, std::vector<Circle>& circles) {
    if (make_part4_scene(name, circles)) return true;
    return make_scene(name, circles);
}

std::vector<std::string> all_scene_names() {
    std::vector<std::string> names = part4_scene_names();
    const std::vector<std::string> starter = scene_names();
    names.insert(names.end(), starter.begin(), starter.end());
    return names;
}
