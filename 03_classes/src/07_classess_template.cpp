#include <type_traits>
#include <iostream>

template <class T,
          class = std::enable_if_t<
              std::is_same_v<std::remove_cv_t<T>, int>  ||
              std::is_same_v<std::remove_cv_t<T>, float>>>
class Box {
    using U = std::remove_cv_t<T>;
    U v;
public:
    explicit Box(U x) : v(x) {}
    U get() const { return v; }
};

int main() {
    Box<int> a{1};
    Box<float> b{2.0f};
    // Box<double> c{3.0}; // ❌ substitution failure = compile error
    std::cout << a.get() << " " << b.get() << "\n";
}