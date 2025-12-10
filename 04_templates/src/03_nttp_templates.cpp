#include <iostream>
using namespace std;

template<int Size>
struct Buffer {
    int data[Size];

    void print() const {
        cout << "Buffer data: ";
        for (int i = 0; i < Size; i++)
            cout << data[i] << " ";
        cout << endl;
    }

    int size() const {
        return Size;
    }
};

template<typename T, int N>
struct ArrayHolder {
    T data[N];

    ArrayHolder(std::initializer_list<T> list) {
        int i = 0;
        for (auto &v : list) {
            if (i < N) data[i++] = v;
        }
        for (; i < N; i++) data[i] = T();
    }

    ArrayHolder() {
        for (int i = 0; i < N; i++) data[i] = T();
    }

    int size() const {
        return N;
    }

    void print() const {
        cout << "ArrayHolder values: ";
        for (int i = 0; i < N; i++)
            cout << data[i] << " ";
        cout << endl;
    }
};

// C++17 deduction guide
template<typename T, typename... Args>
ArrayHolder(T, Args...) -> ArrayHolder<T, 1 + sizeof...(Args)>;

int main() {

    Buffer<5> buf{{1, 2, 3, 4, 5}};
    cout << "Buffer size: " << buf.size() << endl;
    buf.print();

    ArrayHolder arr{5, 10, 12};
    cout << "ArrayHolder (deduced) size: " << arr.size() << endl;
    arr.print();

    ArrayHolder<int, 5> emptyArr;
    cout << "ArrayHolder (default-constructed) size: " << emptyArr.size() << endl;
    emptyArr.print();

    return 0;
}
