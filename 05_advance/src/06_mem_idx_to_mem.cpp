#include <cstdint>
#include <iostream>
#include <unordered_set>
#include  <vector>

using namespace std;

vector<int64_t> make_row_major_strides(const vector<int64_t> & shape) {
    vector<int64_t> strides(shape.size());
    int64_t stride = 1;

    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }

    return strides;
}

size_t offest_of(const vector<int64_t> & idx, const vector<int64_t> & strides) {
    size_t offest = 0;

    for (int i = static_cast<int>(idx.size()) - 1; i >= 0; --i) {
        offest += idx[i] * strides[i];
    }

    return offest;
}

void enumerate_offsets(
    const vector<int64_t>& shape,
    const vector<int64_t>& strides,
    vector<int64_t>& idx,
    int dim,
    unordered_set<int64_t>& offsets,
    bool& overlap
) {
    if (overlap) return; // early stop if overlap already found

    if (dim == (int)shape.size()) {
        int64_t off = 0;
        for (int i = 0; i < (int)idx.size(); i++) {
            off += idx[i] * strides[i];
        }

        if (!offsets.insert(off).second) {
            overlap = true;  // same offset found again
        }
        return;
    }

    for (int64_t i = 0; i < shape[dim]; i++) {
        idx[dim] = i;
        enumerate_offsets(shape, strides, idx, dim + 1, offsets, overlap);
        if (overlap) return; // stop deeper loops too
    }
}

bool has_overlap(const vector<int64_t>& shape, const vector<int64_t>& strides) {
    unordered_set<int64_t> offsets;
    vector<int64_t> idx(shape.size(), 0);

    bool overlap = false;
    enumerate_offsets(shape, strides, idx, 0, offsets, overlap);
    return overlap;
}

int main() {
    vector<int64_t> shape = {2, 3, 4};
    auto strides = make_row_major_strides(shape);

    std::cout << "Strides: ";
    for (auto s: strides) std::cout << s << " ";
    std::cout << "\n";

    vector<int64_t> idx = {1, 2, 3};
    cout << "Offset of (1,2,3): " << offest_of(idx, strides) << "\n";

    cout << boolalpha;
    cout << "Has Overlap: " << has_overlap(shape, strides) << "\n";

    return 0;
}
