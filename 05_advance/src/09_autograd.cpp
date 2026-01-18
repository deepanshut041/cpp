#include <iostream>
#include <memory>

using namespace std;

struct Storage {
    std::shared_ptr<float> data;
    int version = 0;

    explicit Storage(const float v) {
        data = std::shared_ptr<float>(new float(v), [](const float *ptr) {
            delete ptr;
        });
    }
};

struct Node {
    virtual void backward(float grad_out) = 0;
    virtual ~Node() = default;
};

struct TrackedBuffer {
    shared_ptr<Storage> storage_;
    bool requires_grad_ = false;
    float grad_ = 0.0f;
    shared_ptr<Node> grad_fn_ = nullptr;

    static TrackedBuffer create(float v, bool req = false) {
        TrackedBuffer buf;
        buf.storage_ = make_shared<Storage>(v);
        buf.requires_grad_ = req;
        return buf;
    }

    [[nodiscard]] float value() const { return *(storage_->data); }

    void set_value(const float v) const {
        *(storage_->data) = v;
        storage_->version++;
    }

    [[nodiscard]] TrackedBuffer detach() const {
        TrackedBuffer buf;
        buf.storage_ = storage_;
        buf.requires_grad_ = false;
        buf.grad_fn_ = nullptr;
        buf.grad_ = 0.0f;
        return buf;
    }

    [[nodiscard]] TrackedBuffer clone() const {
        TrackedBuffer buf;
        buf.storage_ = make_shared<Storage>(value());
        buf.requires_grad_ = false;
        buf.grad_fn_ = nullptr;
        buf.grad_ = 0.0f;
        return buf;
    }

    void backward(float grad_out = 0.0f) {
        if (!requires_grad_) {return;}
        grad_ += grad_out;
        if (grad_fn_) grad_fn_->backward(grad_out);
    }

};

int main() {
    auto x = TrackedBuffer::create(5, true);
    auto y = x.detach();
    auto z = x.clone();

    y.set_value(99);
    std::cout << "x value after y write: " << x.value() << "\n"; // 99

    z.set_value(42);
    std::cout << "x value after z write: " << x.value() << "\n";

}