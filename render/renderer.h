#pragma once
#include <vector>

// A filled circle with a translucent color.
//
// The image is normalized so that both x and y run over [0, 1]. The center
// (x, y) and the radius r use those units, so on a square image the circle is
// round and r is a fraction of the image width. The color components red,
// green and blue and the opacity alpha all lie in [0, 1].
// Circle order matters because alpha blending is not commutative.
struct Circle {float x,y,r,red,green,blue,alpha;};

// One pixel color with components in [0, 1].
struct Pixel {float r,g,b;};

// A width by height image stored row by row. Pixel (x, y) is pixels[y * width + x].
struct Image {int width,height;std::vector<Pixel> pixels;};

// Sequential reference. Draws the circles in their given order onto a white
// background. Do not change it.
Image render_reference(int width,int height,const std::vector<Circle>& circles);

// Part 3. Parallel renderer that you complete in renderer.cpp. It must match
// render_reference within the tolerance of images_close for every scene.
Image render_parallel(int width,int height,const std::vector<Circle>& circles);

// Part 4. Optimized parallel renderer that you complete in renderer.cpp. It has
// the same contract as render_parallel, but should avoid circle and pixel
// tests that cannot change the result.
Image render_optimized(int width,int height,const std::vector<Circle>& circles);

// Part 4 diagnostics. Runtime schedule selector for schedule(runtime).
// Runtime leaves the current OpenMP runtime schedule untouched, so OMP_SCHEDULE
// still works for the starter demo/benchmark.
enum class RenderSchedule {Runtime=-1,Static=0,Dynamic=1,Guided=2,Auto=3};

// Optional measurements filled by render_parallel_optimized. Preprocessing
// covers image allocation, circle boxes and candidate-list construction;
// render_ms covers the tile loop. Their sum is the total renderer time.
struct RenderStats {
    int tile_size=0;
    long long tile_count=0;
    long long candidate_entries=0;
    long long min_candidates=0;
    long long max_candidates=0;
    double preprocess_ms=0.0;
    double render_ms=0.0;
    // When collect_thread_times is true, these are filled per worker thread
    // with the time spent inside tiles (busy) and the whole loop span.
    bool collect_thread_times=false;
    std::vector<double> thread_busy_ms;
    std::vector<double> thread_span_ms;
};

// Part 4 optimized renderer with explicit configuration. This is an additive
// declaration; the starter signatures above are unchanged. tile_size <= 0
// selects the default (16). chunk_size <= 0 leaves the kind's default chunk.
// Pass stats = nullptr when no measurements are needed.
Image render_parallel_optimized(int width,int height,const std::vector<Circle>& circles,
                                int tile_size,RenderSchedule schedule,int chunk_size,
                                RenderStats* stats=nullptr);

// True when the images have the same size and every color component differs by
// at most the tolerance.
bool images_close(const Image& a,const Image& b,float tolerance=0.0001f);

// Writes the image as a text PPM file. Throws std::runtime_error when the file
// cannot be opened.
void save_ppm(const Image& image,const char* path);
