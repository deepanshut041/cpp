#include <iostream>
#include <memory>
#include <cstring>
#include <stdexcept>

using namespace std;

struct Storage {
    shared_ptr<float> data;
    int64_t size = 0;
    int version = 0;

    Storage(const float* d, int64_t s) : size(s) {
        if (size < 0) throw runtime_error("negative size");
        if (size == 0) {
            data = shared_ptr<float>(nullptr, [](float*) {});
            return;
        }
        data = shared_ptr<float>(new float[size], [](const float* p) { delete[] p; });
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
    virtual void backward(float grad_out) = 0;
    virtual ~Node() = default;
};

struct TrackedTensor {
    shared_ptr<Storage> storage_;
    bool requires_grad_ = false;
    float grad_ = 0.0f;
    shared_ptr<Node> grad_fn_ = nullptr;

    static TrackedTensor create(const float* d, int64_t s, bool req_grad) {
        TrackedTensor t;
        t.storage_ = make_shared<Storage>(d, s);
        t.requires_grad_ = req_grad;
        return t;
    }

    [[nodiscard]] TrackedTensor detach() const {
        TrackedTensor t;
        t.storage_ = storage_;
        t.requires_grad_ = false;
        t.grad_fn_ = nullptr;
        t.grad_ = 0.0f;
        return t;
    }

    [[nodiscard]] TrackedTensor clone() const {
        TrackedTensor t;
        if (!storage_) t.storage_ = make_shared<Storage>(nullptr, 0);
        else t.storage_ = make_shared<Storage>(storage_->data.get(), storage_->size);
        t.requires_grad_ = false;
        t.grad_fn_ = nullptr;
        t.grad_ = 0.0f;
        return t;
    }

    [[nodiscard]] float read(int64_t i) const {
        if (!storage_) throw runtime_error("null storage");
        return storage_->read(i);
    }

    void write(int64_t i, float v) {
        if (!storage_) throw runtime_error("null storage");
        if (storage_.use_count() > 1) {
            auto new_storage = make_shared<Storage>(storage_->data.get(), storage_->size);
            new_storage->version = storage_->version;
            storage_ = new_storage;
        }
        storage_->write(i, v);
    }

    [[nodiscard]] int version() const {
        if (!storage_) return 0;
        return storage_->version;
    }

    [[nodiscard]] int64_t size() const {
        if (!storage_) return 0;
        return storage_->size;
    }

    [[nodiscard]] void* storage_ptr() const {
        return storage_.get();
    }
};

static void print_tensor(const char* name, const TrackedTensor& t) {
    cout << name << ": size=" << t.size()
         << " version=" << t.version()
         << " storage=" << t.storage_ptr() << " data=[";
    for (int64_t i = 0; i < t.size(); i++) {
        if (i) cout << ", ";
        cout << t.read(i);
    }
    cout << "]\n";
}

int main() {
    float init[3] = {1.0f, 2.0f, 3.0f};

    auto a = TrackedTensor::create(init, 3, true);
    auto b = a.detach();
    auto c = a.clone();

    cout << "Initial\n";
    print_tensor("a", a);
    print_tensor("b", b);
    print_tensor("c", c);
    cout << "\n";

    b.write(0, 10.0f);

    cout << "After b.write(0, 10)\n";
    print_tensor("a", a);
    print_tensor("b", b);
    print_tensor("c", c);
    cout << "\n";

    a.write(1, 20.0f);

    cout << "After a.write(1, 20)\n";
    print_tensor("a", a);
    print_tensor("b", b);
    print_tensor("c", c);
    cout << "\n";

    try {
        b.write(100, 1.0f);
    } catch (const exception& e) {
        cout << "Caught expected error on out-of-bounds write: " << e.what() << "\n";
    }

    auto z = TrackedTensor::create(nullptr, 0, false);
    cout << "Zero-size tensor\n";
    print_tensor("z", z);

    try {
        z.write(0, 1.0f);
    } catch (const exception& e) {
        cout << "Caught expected error on zero-size write: " << e.what() << "\n";
    }

    return 0;
}
