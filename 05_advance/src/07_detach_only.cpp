#include <iostream>
#include <memory>
#include <vector>

using namespace std;

struct Buffer {
    shared_ptr<float> data;
    Buffer(float n_data){
        data = shared_ptr<float>(new float(n_data), [](float *ptr) {
            delete ptr;
        });
    }
};

struct Value {
    shared_ptr<Buffer> buf;
    bool requires_grad = false;
    void* grad_fn = nullptr;

    static Value create(float n_data, bool req = false) {
        Value x;
        x.buf = std::make_shared<Buffer>(n_data);
        x.requires_grad = req;
        x.grad_fn = req ? (void*)0x1234 : nullptr;
        return x;
    }

    Value detach() const {
        Value y;
        y.buf = buf;
        y.requires_grad = false;
        y.grad_fn = nullptr;
        return y;
    }

    float& ref() { return *(buf->data.get()); }

};

int main() {
    Value x = Value::create(5.0f, true);
    Value y = x.detach();

    std::cout << "x requires_grad: " << x.requires_grad << "\n";
    std::cout << "y requires_grad: " << y.requires_grad << "\n";

    y.ref() = 99; // modifies shared buffer
    std::cout << "x value: " << x.ref() << "\n"; // prints 99
    return 0;
}