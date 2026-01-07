#include <cstdint>
#include <iostream>
#include <vector>

using namespace std;

vector<int64_t> make_row_major_strides(const vector<int64_t> &shape) {
    vector<int64_t> strides(shape.size());

    int64_t stride = 1;
    for (int i = (int)strides.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }

    return strides;
}

int64_t offset_of(const vector<int64_t> &strides, const vector<int64_t> &idx) {
    int64_t offset = 0;
    for (int i = 0; i < idx.size(); ++i) offset += strides[i] * idx[i];
    return offset;
}
bool is_row_major_contiguous(const vector<int64_t> &shape, const vector<int64_t> &strides) {
    if (shape.size() != strides.size()) return false;

    if (strides.empty()) return true;

    int64_t expected = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        if (shape[i] == 0) return true;
        if (strides[i] != expected) return false;
        expected *= shape[i];
    }

    return true;
}


int main() {
    vector<int64_t> shape{2, 3, 4};
    vector<int64_t> strides = make_row_major_strides(shape);

    cout << "Strides: ";
    for (auto s: strides) cout << s << " ";
    cout << "\n";

    vector<int64_t> idx = {1, 2, 3};
    cout << "Offset of (1,2,3): " << offset_of(strides, idx) << "\n";

    cout << boolalpha;
    cout << "Is contiguous Array: " << is_row_major_contiguous(shape, strides) << "\n";
    return 0;
}