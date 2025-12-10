#include <iostream>
#include <concepts>
#include <functional>
#include <string>

using namespace std;

template <typename T>
concept Hashable = requires(T a) {
    { hash<T>{}(a) } -> std::convertible_to<size_t>;
};

template<Hashable T>
struct Hashbox {
    T value;
};

template<Hashable T>
void print_hash(const Hashbox<T>& hb) {
    cout << "hash(" << hb.value << ") = " << hash<T>{}(hb.value) << '\n';
}

struct NoHash {};

int main() {
    static_assert(Hashable<int>);
    static_assert(Hashable<string>);
    static_assert(!Hashable<NoHash>);

    Hashbox<int> hi{42};
    Hashbox<string> hs{string("hello")};

    print_hash(hi);
    print_hash(hs);

    // Hashbox<NoHash> hn{NoHash{}};   // must fail to compile if uncommented

    return 0;
}
