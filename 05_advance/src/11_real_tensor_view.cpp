#include <cstring>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace std;

static vector<int64_t> make_row_strides(const vector<int64_t>& shape) {
    vector<int64_t> strides(shape.size(), 0);
    if (shape.empty()) return strides;

    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

static bool is_row_major_contiguous(const vector<int64_t>& shape, const vector<int64_t>& strides) {
    if (shape.size() != strides.size()) return false;
    if (shape.empty()) return true;

    int64_t expected = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        if (shape[i] == 0) return true;
        if (strides[i] != expected) return false;
        expected *= shape[i];
    }
    return true;
}

static int64_t dot_strides(const vector<int64_t>& strides, const vector<int64_t>& idx) {
    if (idx.size() != strides.size()) throw runtime_error("index rank != tensor rank");
    int64_t off = 0;
    for (size_t i = 0; i < idx.size(); ++i) off += strides[i] * idx[i];
    return off;
}

struct Storage {
    shared_ptr<float> data;
    int64_t size;
    int version = 0;

    Storage(const float* d, int64_t s) : size(s) {
        if (size < 0) throw runtime_error("negative storage size");
        if (size == 0) {
            data = shared_ptr<float>(nullptr, [](float*) {});
            return;
        }

        data = shared_ptr<float>(new float[static_cast<size_t>(size)], [](const float* ptr) { delete[] ptr; });

        if (d) memcpy(data.get(), d, static_cast<size_t>(size) * sizeof(float));
        else memset(data.get(), 0, static_cast<size_t>(size) * sizeof(float));
    }

    [[nodiscard]] float read(int64_t i) const {
        if (i < 0 || i >= size) throw out_of_range("read out of bounds");
        return data.get()[i];
    }

    void write(int64_t i, float v) {
        if (i < 0 || i >= size) throw out_of_range("write out of bounds");
        data.get()[i] = v;
        version += 1;
    }
};

struct Node {
    virtual void backward(float /*out_grad*/) = 0;
    virtual ~Node() = default;
};

struct Tensor {
    shared_ptr<Storage> storage_;
    vector<int64_t> shape_;
    vector<int64_t> strides_;
    int64_t offset_ = 0;
    bool requires_grad_ = false;
    shared_ptr<Node> grad_fn_;
    shared_ptr<Tensor> grad_;

    static Tensor create(const float* d, const vector<int64_t>& shape, bool requires_grad = false) {
        Tensor t;
        int64_t size = 1;
        for (int64_t dim : shape) {
            if (dim < 0) throw runtime_error("negative dim in shape");
            size *= dim;
        }
        t.storage_ = make_shared<Storage>(d, size);
        t.shape_ = shape;
        t.strides_ = make_row_strides(shape);
        t.offset_ = 0;
        t.requires_grad_ = requires_grad;
        return t;
    }

    [[nodiscard]] int64_t numel() const {
        int64_t n = 1;
        for (int64_t dim : shape_) n *= dim;
        return n;
    }

    [[nodiscard]] bool is_contiguous_row_major() const {
        return is_row_major_contiguous(shape_, strides_);
    }

    [[nodiscard]] float at(const vector<int64_t>& idx) const {
        if (idx.size() != shape_.size()) throw runtime_error("at(): index rank != tensor rank");
        for (size_t i = 0; i < idx.size(); ++i) {
            if (idx[i] < 0 || idx[i] >= shape_[i]) throw out_of_range("at(): index out of bounds");
        }
        int64_t lin = offset_ + dot_strides(strides_, idx);
        return storage_->read(lin);
    }

    [[nodiscard]] float at(const initializer_list<int64_t> idx) const {
        return at(vector<int64_t>(idx));
    }

    void set(const vector<int64_t>& idx, const float v) const {
        if (idx.size() != shape_.size()) throw runtime_error("set(): index rank != tensor rank");
        for (size_t i = 0; i < idx.size(); ++i) {
            if (idx[i] < 0 || idx[i] >= shape_[i]) throw out_of_range("set(): index out of bounds");
        }
        const int64_t lin = offset_ + dot_strides(strides_, idx);
        storage_->write(lin, v);
    }

    void set(const initializer_list<int64_t> idx, const float v) const {
        set(vector<int64_t>(idx), v);
    }

    void view_reshape(const vector<int64_t>& new_shape) {
        if (!is_contiguous_row_major()) {
            throw runtime_error("view_reshape requires contiguous row-major tensor");
        }
        int64_t new_numel = 1;
        for (int64_t dim : new_shape) {
            if (dim < 0) throw runtime_error("negative dim in new_shape");
            new_numel *= dim;
        }
        if (new_numel != numel()) throw runtime_error("reshape numel mismatch");
        shape_ = new_shape;
        strides_ = make_row_strides(new_shape);
    }

    [[nodiscard]] Tensor transpose(const int64_t dim0, const int64_t dim1) const {
        if (const auto r = static_cast<int64_t>(shape_.size()); dim0 < 0 || dim1 < 0 || dim0 >= r || dim1 >= r) throw runtime_error("transpose dims out of range");

        Tensor t = *this; // copy view metadata (shared storage)
        swap(t.shape_[dim0], t.shape_[dim1]);
        swap(t.strides_[dim0], t.strides_[dim1]);
        t.grad_fn_ = nullptr;
        t.grad_ = nullptr;
        return t;
    }

    [[nodiscard]] Tensor slice(int64_t dim, int64_t start, int64_t stop, int64_t step) const {
        if (step == 0) throw runtime_error("slice step cannot be 0");
        if (const auto r = static_cast<int64_t>(shape_.size()); dim < 0 || dim >= r) throw runtime_error("slice dim out of range");

        const int64_t n = shape_[dim];
        if (n < 0) throw runtime_error("negative dim size");
        Tensor t = *this;
        t.grad_fn_ = nullptr;
        t.grad_ = nullptr;

        if (n == 0) {
            t.shape_[dim] = 0;
            return t;
        }

        auto clamp = [](const int64_t x, const int64_t lo, const int64_t hi) -> int64_t {
            if (x < lo) return lo;
            if (x > hi) return hi;
            return x;
        };

        if (step > 0) {
            if (start < 0) start += n;
            if (stop < 0) stop += n;

            start = clamp(start, 0, n);
            stop = clamp(stop, 0, n);

            int64_t len = 0;
            if (stop > start) len = (stop - start + step - 1) / step;

            t.offset_ = offset_ + start * strides_[dim];
            t.shape_[dim] = len;
            t.strides_[dim] *= step;
            return t;
        }
        if (start < 0) start += n;
        if (stop < 0) stop += n;
        start = clamp(start, 0, n - 1);
        stop = clamp(stop, -1, n - 1);

        int64_t step_abs = -step;
        int64_t len = 0;
        if (start > stop) len = (start - stop + step_abs - 1) / step_abs;

        t.offset_ = offset_ + start * strides_[dim];
        t.shape_[dim] = len;
        t.strides_[dim] *= step; // negative stride
        return t;
    }
};

static void print_vec(const string& name, const vector<int64_t>& v) {
    cout << name << " = [";
    for (size_t i = 0; i < v.size(); ++i) cout << v[i] << (i + 1 < v.size() ? ", " : "");
    cout << "]\n";
}

static void print_tensor_meta(const string& title, const Tensor& x) {
    cout << title << "\n";
    print_vec("shape", x.shape_);
    print_vec("strides", x.strides_);
    cout << "offset = " << x.offset_ << "\n";
    cout << "numel = " << x.numel() << "\n";
    cout << "is_contiguous = " << (x.is_contiguous_row_major() ? "true" : "false") << "\n";
    cout << "storage size = " << x.storage_->size << "\n";
    cout << "storage version = " << x.storage_->version << "\n\n";
}

int main() {
    vector<float> host(24);
    for (int i = 0; i < 24; ++i) host[i] = static_cast<float>(i);

    Tensor base = Tensor::create(host.data(), {2, 3, 4});

    print_tensor_meta("Base tensor:", base);

    cout << "Base at(0,1,2) = " << base.at({0, 1, 2}) << "\n"; // expected 0*12 + 1*4 + 2 = 6
    cout << "Base at(1,2,3) = " << base.at({1, 2, 3}) << "\n\n"; // expected 1*12 + 2*4 + 3 = 23

    // Transpose view: swap dims 1 and 2 (shape 2x4x3), strides swapped
    Tensor tr = base.transpose(1, 2);
    print_tensor_meta("Transpose view (dim1 <-> dim2):", tr);

    // Slice view: take every 2nd element along last dim: base[:,:,1:4:2] -> last dim length 2
    Tensor sl = base.slice(2, 1, 4, 2);
    print_tensor_meta("Slice view (dim2: start=1, stop=4, step=2):", sl);

    // ---- Aliasing proof: write through transpose view ----
    // base(0,1,2) == 6. In transpose (dim1<->dim2), that same element is at tr(0,2,1)
    cout << "Before: base(0,1,2) = " << base.at({0, 1, 2}) << ", tr(0,2,1) = " << tr.at({0, 2, 1}) << "\n";
    tr.set({0, 2, 1}, 999.0f);
    cout << "After tr.set(0,2,1)=999:\n";
    cout << "  base(0,1,2) = " << base.at({0, 1, 2}) << "  (should be 999)\n";
    cout << "  storage version = " << base.storage_->version << "\n\n";

    // ---- Aliasing proof: write through slice view ----
    // sl has shape [2,3,2] where sl(i,j,k) maps to base(i,j, 1 + k*2)
    cout << "Before: sl(1,2,1) = " << sl.at({1, 2, 1}) << " maps to base(1,2,3) = " << base.at({1, 2, 3}) << "\n";
    sl.set({1, 2, 1}, 1234.0f);
    cout << "After sl.set(1,2,1)=1234:\n";
    cout << "  base(1,2,3) = " << base.at({1, 2, 3}) << "  (should be 1234)\n";
    cout << "  storage version = " << base.storage_->version << "\n\n";

    // ---- reshape demo: only allowed on contiguous ----
    Tensor flat = base;              // base is contiguous
    flat.view_reshape({24});
    print_tensor_meta("Reshaped contiguous view -> [24]:", flat);

    // Uncomment to see it fail:
    // Tensor bad = base.transpose(0, 2);
    // bad.view_reshape({24}); // throws

    return 0;
}
