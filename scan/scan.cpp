// Part 2: reference scan and repeat compaction, followed by the two functions to complete.
#include "scan.h"
#include <cstddef>
#include <memory>
#include <vector>

// Reference implementations. Every parallel result is compared against these.
std::vector<int> exclusive_scan_serial(const std::vector<int>& input) {
    std::vector<int> result(input.size());
    int sum=0;
    for (size_t i=0;i<input.size();++i) {result[i]=sum;sum+=input[i];}
    return result;
}
std::vector<int> find_repeats_serial(const std::vector<int>& input) {
    std::vector<int> indices;
    for (size_t i=0;i+1<input.size();++i)
        if(input[i]==input[i+1]) indices.push_back(static_cast<int>(i));
    return indices;
}

// Your implementations go here. You may add helper functions above them.
namespace {

// Smallest power of two greater than or equal to n (n > 0).
std::size_t next_power_of_two(std::size_t n) {
    std::size_t power = 1;
    while (power < n) power <<= 1;
    return power;
}

}  // namespace

std::vector<int> exclusive_scan_parallel(const std::vector<int>& input) {
    const std::size_t n = input.size();
    if (n == 0) return {};
    if (n == 1) return {0};

    // Pad with zero, the identity of addition, to the next power of two. The
    // padding is invisible in the output because the result is truncated to n.
    const std::size_t N = next_power_of_two(n);
    std::unique_ptr<int[]> tree(new int[N]);

    #pragma omp parallel
    {
        // Copy the input and zero the padding in parallel. Every index is
        // written by exactly one thread.
        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < N; ++i) tree[i] = i < n ? input[i] : 0;

        // Up-sweep: tree[node] becomes the sum of its subtree. At one level the
        // end indices k + 2*d - 1 are disjoint, so updates need no atomics.
        // The implicit barrier after each omp for separates the levels.
        for (std::size_t d = 1; d < N; d <<= 1) {
            #pragma omp for schedule(static)
            for (std::size_t k = 0; k < N; k += 2 * d)
                tree[k + 2 * d - 1] += tree[k + d - 1];
        }

        // Turn the inclusive root sum into the exclusive identity.
        #pragma omp single
        tree[N - 1] = 0;

        // Down-sweep: replace each node with the exclusive prefix for its
        // subtree. The two indices touched per block are disjoint across k.
        for (std::size_t d = N >> 1; d > 0; d >>= 1) {
            #pragma omp for schedule(static)
            for (std::size_t k = 0; k < N; k += 2 * d) {
                const int left = tree[k + d - 1];
                tree[k + d - 1] = tree[k + 2 * d - 1];
                tree[k + 2 * d - 1] += left;
            }
        }
    }

    // Range construction copies without a zero-initialization pass.
    return std::vector<int>(tree.get(), tree.get() + n);
}

std::vector<int> find_repeats_parallel(const std::vector<int>& input) {
    const std::size_t n = input.size();
    if (n < 2) return {};
    const std::size_t m = n - 1;

    // Step 1: flags[i] is 1 exactly when position i is a repeat. Reading
    // input[i + 1] is safe because i < n - 1.
    std::vector<int> flags(m);
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < m; ++i)
        flags[i] = input[i] == input[i + 1] ? 1 : 0;

    // Step 2: exclusive prefix sums give each flagged position its slot.
    const std::vector<int> offsets = exclusive_scan_parallel(flags);

    // Step 3: the last offset plus the last flag is the exact output length.
    const std::size_t total =
        static_cast<std::size_t>(offsets[m - 1] + flags[m - 1]);
    std::vector<int> result(total);

    // Step 4: scatter. Offsets increase by one at every flagged position, so
    // two iterations never write the same slot; ascending i gives ascending
    // output indices.
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < m; ++i)
        if (flags[i]) result[static_cast<std::size_t>(offsets[i])] = static_cast<int>(i);

    return result;
}
