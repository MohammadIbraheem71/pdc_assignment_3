#pragma once
// Deterministic Part 4 scenes for the optimization study. These live in their
// own files so the starter scenes and their generators stay untouched. Every
// scene uses a fixed Lcg seed, so it is identical on every run and machine.
#include "renderer.h"
#include <string>
#include <vector>

// Part 4 scene names:
//   p4_sparse 2000 tiny circles spread uniformly; most pixels are empty.
//   p4_heavy  400 large translucent circles covering most of the image.
//   p4_corner 700 circles packed into the top-left corner plus 60 scattered.
//   p4_few    8 medium circles; culling has little structure but setup is small.
//   p4_one    1 huge circle; every tile has the same single candidate.
//   p4_huge   32 huge circles; every tile contains every circle, so no culling.
std::vector<std::string> part4_scene_names();

// Fills circles with the named Part 4 scene. Returns false and leaves circles
// unchanged when the name is not a Part 4 scene.
bool make_part4_scene(const std::string& name, std::vector<Circle>& circles);

// Tries the Part 4 scenes first, then the starter scenes. This lets the Part 4
// drivers accept the existing names (dense, sparse, overlap, clustered) too.
bool make_any_scene(const std::string& name, std::vector<Circle>& circles);

// Part 4 names followed by the starter scene names.
std::vector<std::string> all_scene_names();
