#include <cstdint>
#include <iostream>
#include <vector>

using namespace std;


std::vector<int64_t> make_row_major_strides(const vector<int64_t> &shape) {
    std::vector<int64_t> strides(shape.size());

    int64_t running = 1;
    for (int i = (int) shape.size() - 1; i >= 0; --i) {
        strides[i] = running;
        running *= shape[i];
    }
    return strides;
};

int64_t offset_of(const vector<int64_t> &idx, const vector<int64_t> &strides) {
    if (idx.size() != strides.size()) throw std::runtime_error("rank mismatch");

    int64_t offset = 0;
    for (int i = 0; i < idx.size(); ++i) offset += strides[i] * idx[i];
    return offset;
}

int main() {
    vector<int64_t> shape = {3, 16, 16};
    auto strides = make_row_major_strides(shape);

    std::cout << "Strides: ";
    for (auto s: strides) std::cout << s << " ";
    std::cout << "\n";

    std::vector<int64_t> idx = {1, 2, 3};
    std::cout << "Offset of (1,2,3): " << offset_of(idx, strides) << "\n";

    return 0;
}
