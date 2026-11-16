// Part 2 report aid: prints the array after every up-sweep and down-sweep level
// of the Blelloch exclusive scan for n = 16.
//
// Build/run from the repository root:
//   ./scan/run_stages.sh
#include <cstdio>
#include <vector>

namespace {

void print_array(const char* label, const std::vector<int>& tree) {
    std::printf("%-22s [", label);
    for (std::size_t i = 0; i < tree.size(); ++i)
        std::printf("%s%d", i == 0 ? "" : ", ", tree[i]);
    std::printf("]\n");
}

}  // namespace

int main() {
    const std::vector<int> expected = {0, 1, 3, 6, 10, 11, 13, 16,
                                       20, 21, 23, 26, 30, 31, 33, 36};
    std::vector<int> tree = {1, 2, 3, 4, 1, 2, 3, 4,
                             1, 2, 3, 4, 1, 2, 3, 4};
    const std::size_t n = tree.size();

    std::printf("n = %zu, input = [1,2,3,4] repeated four times\n", n);
    print_array("initial", tree);

    for (std::size_t d = 1; d < n; d <<= 1) {
        for (std::size_t k = 0; k < n; k += 2 * d)
            tree[k + 2 * d - 1] += tree[k + d - 1];
        char label[64];
        std::snprintf(label, sizeof(label), "up-sweep d=%zu", d);
        print_array(label, tree);
    }

    tree[n - 1] = 0;
    print_array("root set to 0", tree);

    for (std::size_t d = n >> 1; d > 0; d >>= 1) {
        for (std::size_t k = 0; k < n; k += 2 * d) {
            const int left = tree[k + d - 1];
            tree[k + d - 1] = tree[k + 2 * d - 1];
            tree[k + 2 * d - 1] += left;
        }
        char label[64];
        std::snprintf(label, sizeof(label), "down-sweep d=%zu", d);
        print_array(label, tree);
    }

    print_array("final", tree);
    const bool ok = tree == expected;
    std::printf("\nexpected exclusive scan: [");
    for (std::size_t i = 0; i < expected.size(); ++i)
        std::printf("%s%d", i == 0 ? "" : ", ", expected[i]);
    std::printf("]\n");
    std::printf("STAGED SCAN: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
