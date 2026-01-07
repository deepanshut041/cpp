#include <iostream>
#include <memory>

using namespace std;

struct Buffer {
    shared_ptr<float> data;
    Buffer(float _data) {
        data = shared_ptr<float>(new float(_data), [](float *ptr) {
            delete ptr;
        });
    }
};

struct Value {
    shared_ptr<Buffer> buf_;
    bool requires_grad = false;
    void* grad_fn = nullptr;

    static Value create(float _data, bool req = false) {
        Value v;
        v.buf_ = make_shared<Buffer>(_data);
        v.requires_grad = req;
        v.grad_fn = req ? (void*)0x1234 : nullptr;
        return v;
    }

    Value detach() const {
        Value v;
        v.buf_ = buf_;
        v.requires_grad = false;
        v.grad_fn = nullptr;
        return v;
    }

    Value clone() const {
        Value v;
        v.buf_ = make_shared<Buffer>(*(buf_->data));
        v.requires_grad = requires_grad;
        v.grad_fn = grad_fn;
        return v;
    }

    float& ref() { return *(buf_->data.get()); }

    void print(string name) {
        cout << "Name: " << name << "\n" << "___________________" << endl;
        cout << "Value of float: " << ref() << endl;
        cout << "Address of float: " << buf_->data.get() << endl;
        cout << "Requires Grad: " << requires_grad << endl;
        cout << "Grad Fn: " << grad_fn << endl << endl;
    }
};

int main() {
    Value x = Value::create(5.0f, true);
    x.print("X");

    Value y = x.detach();
    y.ref() = 99;

    Value z = x.clone();

    x.print("X");
    y.print("Y");
    z.print("Z");

    return 0;
}