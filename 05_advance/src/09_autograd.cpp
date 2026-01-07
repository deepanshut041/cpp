#include <iostream>
#include <memory>

using namespace std;

struct Storage {
    shared_ptr<float> data;
    int version = 0;
    Storage(float v) {
        data = shared_ptr<float>(new float(v), [] (float* p) {
            delete p;
        });
    }
};

struct Node {
    virtual void backward(float grad_out) = 0;
    virtual ~Node() = default;
};

struct Vector {
    shared_ptr<Storage> storage_;
    bool requires_grad_ = false;
    float grad_ = 0.0f;
    shared_ptr<Node> grad_fn_ = nullptr;

    static Vector create(float v, bool req=false) {
        Vector vec;
        vec.storage_ = make_shared<Storage>(v);
        vec.requires_grad_= req;

        return vec;
    }

    float value() const{ return *(storage_->data); }

    void set_value(float v) {
        *(storage_->data) = v;
        storage_->version++;
    }

    Vector detach() const {
        Vector vec;
        vec.storage_ = storage_;  // deep copy
        vec.requires_grad_ = false;
        vec.grad_fn_ = nullptr;
        vec.grad_ = 0.0f;
        return vec;
    }

    Vector clone() const {
        Vector y;
        y.storage_ = std::make_shared<Storage>(value());  // deep copy
        y.requires_grad_ = false;
        y.grad_fn_ = nullptr;
        y.grad_ = 0.0f;
        return y;
    }

    void backward(float grad_out = 1.0f) {
        if (!requires_grad_) return;
        grad_ += grad_out;
        if (grad_fn_) grad_fn_->backward(grad_out);
    }
};

int main() {
    cout << "Hello world!" << endl;
    auto x = Vector::create(5, true);
    auto y = x.detach();
    auto z = x.clone();

    y.set_value(99);
    std::cout << "x value after y write: " << x.value() << "\n"; // 99

    z.set_value(42);
    std::cout << "x value after z write: " << x.value() << "\n"; // still 99
    return 0;
}