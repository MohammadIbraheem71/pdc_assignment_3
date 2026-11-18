// Parts 3 and 4: point in circle coverage and ordered alpha blending.
#include "renderer.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <omp.h>
#include <stdexcept>
namespace {

// Applies one circle to one pixel: new = alpha * color + (1 minus alpha) * old,
// computed separately for the red, green and blue components.
void blend(Pixel& p,const Circle& c) {
    p.r=c.alpha*c.red+(1-c.alpha)*p.r;
    p.g=c.alpha*c.green+(1-c.alpha)*p.g;
    p.b=c.alpha*c.blue+(1-c.alpha)*p.b;
}

// A pixel is covered when its center lies inside the circle, boundary included.
// The center of pixel (x, y) is ((x + 0.5) / w, (y + 0.5) / h) in the
// normalized coordinates of the image.
bool covers(const Circle& c,int x,int y,int w,int h) {
    float px=(x+0.5f)/w, py=(y+0.5f)/h;
    float dx=px-c.x,dy=py-c.y;
    return dx*dx+dy*dy<=c.r*c.r;
}

// Conservative pixel bounding box of a circle. A pixel center lies at
// (x+0.5)/w, so the covered x range is w*(cx-r)-0.5 .. w*(cx+r)-0.5. The box
// is rounded outward and then padded by one extra pixel on every side, because
// float rounding in covers() may put a pixel center exactly on the edge.
struct Box {int min_x,min_y,max_x,max_y;bool empty;};

Box circle_box(const Circle& c,int w,int h) {
    Box box{0,0,-1,-1,true};
    if(w<=0||h<=0) return box;
    const double r=std::fabs(static_cast<double>(c.r));
    const double cx=static_cast<double>(c.x), cy=static_cast<double>(c.y);
    const double min_x=std::floor(static_cast<double>(w)*(cx-r)-0.5)-1.0;
    const double max_x=std::ceil (static_cast<double>(w)*(cx+r)-0.5)+1.0;
    const double min_y=std::floor(static_cast<double>(h)*(cy-r)-0.5)-1.0;
    const double max_y=std::ceil (static_cast<double>(h)*(cy+r)-0.5)+1.0;
    if(!std::isfinite(min_x)||!std::isfinite(max_x)||
       !std::isfinite(min_y)||!std::isfinite(max_y)) return box;
    if(max_x<0.0||max_y<0.0||min_x>w-1||min_y>h-1) return box;
    box.min_x=static_cast<int>(std::max(0.0,min_x));
    box.max_x=static_cast<int>(std::min(static_cast<double>(w-1),max_x));
    box.min_y=static_cast<int>(std::max(0.0,min_y));
    box.max_y=static_cast<int>(std::min(static_cast<double>(h-1),max_y));
    box.empty=box.min_x>box.max_x||box.min_y>box.max_y;
    return box;
}

void apply_schedule(RenderSchedule schedule,int chunk_size) {
    const int chunk=chunk_size>0?chunk_size:0;
    switch(schedule) {
        case RenderSchedule::Static:  omp_set_schedule(omp_sched_static,chunk);break;
        case RenderSchedule::Dynamic: omp_set_schedule(omp_sched_dynamic,chunk);break;
        case RenderSchedule::Guided:  omp_set_schedule(omp_sched_guided,chunk);break;
        case RenderSchedule::Auto:    omp_set_schedule(omp_sched_auto,chunk);break;
        case RenderSchedule::Runtime: break;
    }
}
}

// Reference renderer. It visits the circles in order and, for each circle,
// every pixel of the image, so its cost is proportional to circles times pixels.
Image render_reference(int width,int height,const std::vector<Circle>& circles) {
    Image image{width,height,std::vector<Pixel>(width*height,Pixel{1,1,1})};
    for (const Circle& c : circles)
        for (int y=0;y<height;++y)
            for (int x=0;x<width;++x)
                if(covers(c,x,y,width,height)) blend(image.pixels[y*width+x],c);
    return image;
}

// Part 3. Parallel baseline, kept unchanged for the Part 4 comparison.
// Rows are independent, so each worker owns a disjoint set of pixels. Inside a
// pixel the circles are applied in their original input order, which is
// required because alpha blending is not commutative.
Image render_parallel(int width,int height,const std::vector<Circle>& circles) {
    Image image{width,height,std::vector<Pixel>(width*height,Pixel{1,1,1})};
    #pragma omp parallel for schedule(runtime)
    for (int y=0;y<height;++y) {
        for (int x=0;x<width;++x) {
            Pixel p{1,1,1};
            for (const Circle& c : circles)
                if(covers(c,x,y,width,height)) blend(p,c);
            image.pixels[static_cast<std::size_t>(y)*width+x]=p;
        }
    }
    return image;
}

// Part 4. Ordered per-tile candidate lists. The circle bounding box is only a
// filter: every pixel still uses covers() and blend() in ascending circle index.
// The new entry point keeps the configuration explicit for the benchmark driver;
// render_optimized() below is the starter-facing wrapper.
Image render_parallel_optimized(int width,int height,const std::vector<Circle>& circles,
                                int tile_size,RenderSchedule schedule,int chunk_size,
                                RenderStats* stats) {
    if(tile_size<=0) tile_size=16;
    const double preprocess_start=stats?omp_get_wtime():0.0;

    Image image{width,height,std::vector<Pixel>(
        static_cast<std::size_t>(std::max(0,width))*static_cast<std::size_t>(std::max(0,height)),
        Pixel{1,1,1})};
    if(stats) {
        stats->tile_size=tile_size;
        stats->tile_count=0;
        stats->candidate_entries=0;
        stats->min_candidates=0;
        stats->max_candidates=0;
        stats->preprocess_ms=0.0;
        stats->render_ms=0.0;
        stats->thread_busy_ms.clear();
        stats->thread_span_ms.clear();
    }
    if(width<=0||height<=0) {
        if(stats) stats->preprocess_ms=(omp_get_wtime()-preprocess_start)*1000.0;
        return image;
    }

    const int tiles_x=(width+tile_size-1)/tile_size;
    const int tiles_y=(height+tile_size-1)/tile_size;
    const long long tile_count=static_cast<long long>(tiles_x)*tiles_y;
    const int circle_count=static_cast<int>(circles.size());

    // 1. Conservative boxes, one independent box per circle.
    std::vector<Box> boxes(static_cast<std::size_t>(circle_count));
    #pragma omp parallel for schedule(static)
    for(int i=0;i<circle_count;++i) boxes[static_cast<std::size_t>(i)]=circle_box(
        circles[static_cast<std::size_t>(i)],width,height);

    // 2. Count candidates per tile. For normal tile counts each worker counts
    // into a private array to avoid atomics; the reduction is deterministic.
    std::vector<int> counts(static_cast<std::size_t>(tile_count),0);
    if(tile_count<=65536 && circle_count>0) {
        const int workers=omp_get_max_threads();
        std::vector<std::vector<int>> local(
            static_cast<std::size_t>(workers),std::vector<int>(
                static_cast<std::size_t>(tile_count),0));
        #pragma omp parallel
        {
            std::vector<int>& mine=local[static_cast<std::size_t>(omp_get_thread_num())];
            #pragma omp for schedule(static)
            for(int i=0;i<circle_count;++i) {
                const Box& b=boxes[static_cast<std::size_t>(i)];
                if(b.empty) continue;
                const int tx0=b.min_x/tile_size,tx1=b.max_x/tile_size;
                const int ty0=b.min_y/tile_size,ty1=b.max_y/tile_size;
                for(int ty=ty0;ty<=ty1;++ty) {
                    const long long base=static_cast<long long>(ty)*tiles_x;
                    for(int tx=tx0;tx<=tx1;++tx) ++mine[static_cast<std::size_t>(base+tx)];
                }
            }
        }
        #pragma omp parallel for schedule(static)
        for(long long t=0;t<tile_count;++t) {
            long long sum=0;
            for(int w=0;w<workers;++w) sum+=local[static_cast<std::size_t>(w)]
                [static_cast<std::size_t>(t)];
            counts[static_cast<std::size_t>(t)]=static_cast<int>(sum);
        }
    } else {
        for(int i=0;i<circle_count;++i) {
            const Box& b=boxes[static_cast<std::size_t>(i)];
            if(b.empty) continue;
            const int tx0=b.min_x/tile_size,tx1=b.max_x/tile_size;
            const int ty0=b.min_y/tile_size,ty1=b.max_y/tile_size;
            for(int ty=ty0;ty<=ty1;++ty) {
                const long long base=static_cast<long long>(ty)*tiles_x;
                for(int tx=tx0;tx<=tx1;++tx) ++counts[static_cast<std::size_t>(base+tx)];
            }
        }
    }

    // 3. Prefix sum over tiles gives the CSR offsets.
    std::vector<std::size_t> offsets(static_cast<std::size_t>(tile_count)+1,0);
    long long minimum=LLONG_MAX,maximum=0;
    for(long long t=0;t<tile_count;++t) {
        const long long c=counts[static_cast<std::size_t>(t)];
        offsets[static_cast<std::size_t>(t+1)]=
            offsets[static_cast<std::size_t>(t)]+static_cast<std::size_t>(c);
        minimum=std::min(minimum,c);
        maximum=std::max(maximum,c);
    }
    if(tile_count==0) minimum=0;

    // 4. Fill in ascending circle order. Because the outer loop visits circles
    // 0..N-1, every tile's slice of the flat candidate array is ascending.
    std::vector<int> candidates(offsets[static_cast<std::size_t>(tile_count)]);
    std::vector<std::size_t> cursor(offsets.begin(),offsets.end()-1);
    for(int i=0;i<circle_count;++i) {
        const Box& b=boxes[static_cast<std::size_t>(i)];
        if(b.empty) continue;
        const int tx0=b.min_x/tile_size,tx1=b.max_x/tile_size;
        const int ty0=b.min_y/tile_size,ty1=b.max_y/tile_size;
        for(int ty=ty0;ty<=ty1;++ty) {
            const long long base=static_cast<long long>(ty)*tiles_x;
            for(int tx=tx0;tx<=tx1;++tx) {
                const std::size_t t=static_cast<std::size_t>(base+tx);
                candidates[cursor[t]++]=i;
            }
        }
    }

    if(stats) {
        stats->tile_count=tile_count;
        stats->candidate_entries=static_cast<long long>(
            offsets[static_cast<std::size_t>(tile_count)]);
        stats->min_candidates=tile_count==0?0:minimum;
        stats->max_candidates=tile_count==0?0:maximum;
        stats->preprocess_ms=(omp_get_wtime()-preprocess_start)*1000.0;
    }

    // 5. Render whole tiles. Tiles own disjoint pixels, so no synchronization is
    // needed. The CSR order is the original circle order.
    auto render_tile=[&](long long t) {
        const int ty=static_cast<int>(t/tiles_x);
        const int tx=static_cast<int>(t%tiles_x);
        const int x0=tx*tile_size,x1=std::min(x0+tile_size,width);
        const int y0=ty*tile_size,y1=std::min(y0+tile_size,height);
        const std::size_t begin=offsets[static_cast<std::size_t>(t)];
        const std::size_t end=offsets[static_cast<std::size_t>(t+1)];
        for(int y=y0;y<y1;++y) {
            Pixel* row=&image.pixels[static_cast<std::size_t>(y)*width];
            for(int x=x0;x<x1;++x) {
                Pixel p{1,1,1};
                for(std::size_t k=begin;k<end;++k) {
                    const Circle& c=circles[static_cast<std::size_t>(candidates[k])];
                    if(covers(c,x,y,width,height)) blend(p,c);
                }
                row[x]=p;
            }
        }
    };

    apply_schedule(schedule,chunk_size);
    const double render_start=stats?omp_get_wtime():0.0;
    if(stats&&stats->collect_thread_times) {
        const int workers=omp_get_max_threads();
        stats->thread_busy_ms.assign(static_cast<std::size_t>(workers),0.0);
        stats->thread_span_ms.assign(static_cast<std::size_t>(workers),0.0);
        #pragma omp parallel
        {
            const int tid=omp_get_thread_num();
            double busy=0.0;
            const double start=omp_get_wtime();
            #pragma omp for schedule(runtime)
            for(long long t=0;t<tile_count;++t) {
                const double a=omp_get_wtime();
                render_tile(t);
                busy+=omp_get_wtime()-a;
            }
            // omp_get_wtime() returns seconds; the stats fields are milliseconds.
            stats->thread_busy_ms[static_cast<std::size_t>(tid)]=busy*1000.0;
            stats->thread_span_ms[static_cast<std::size_t>(tid)]=
                (omp_get_wtime()-start)*1000.0;
        }
    } else {
        #pragma omp parallel for schedule(runtime)
        for(long long t=0;t<tile_count;++t) render_tile(t);
    }
    if(stats) stats->render_ms=(omp_get_wtime()-render_start)*1000.0;
    return image;
}

// Starter-facing Part 4 wrapper. RENDER_TILE changes the tile size; the
// schedule is left to OMP_SCHEDULE / omp_set_schedule because the loop uses
// schedule(runtime).
Image render_optimized(int width,int height,const std::vector<Circle>& circles) {
    int tile_size=16;
    if(const char* text=std::getenv("RENDER_TILE")) {
        const int value=std::atoi(text);
        if(value>0) tile_size=value;
    }
    return render_parallel_optimized(width,height,circles,tile_size,
                                     RenderSchedule::Runtime,0,nullptr);
}

// Compares two images component by component against the tolerance.
bool images_close(const Image& a,const Image& b,float t) {
    if(a.width!=b.width||a.height!=b.height||a.pixels.size()!=b.pixels.size())return false;
    for(size_t i=0;i<a.pixels.size();++i) {
        const Pixel& x=a.pixels[i];const Pixel& y=b.pixels[i];
        if(std::abs(x.r-y.r)>t||std::abs(x.g-y.g)>t||std::abs(x.b-y.b)>t)return false;
    }
    return true;
}

// Writes the image as a text PPM. Each component is clamped to [0, 1] and
// scaled to 0 through 255.
void save_ppm(const Image& image,const char* path) {
    std::ofstream file(path);
    if(!file)throw std::runtime_error("Cannot open PPM output file");
    file<<"P3\n"<<image.width<<" "<<image.height<<"\n255\n";
    for(const Pixel& p:image.pixels) {
        auto ch=[](float x){return int(std::round(std::max(0.0f,std::min(1.0f,x))*255));};
        file<<ch(p.r)<<" "<<ch(p.g)<<" "<<ch(p.b)<<"\n";
    }
}
