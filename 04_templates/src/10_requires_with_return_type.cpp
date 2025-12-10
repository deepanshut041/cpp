#include <iostream>
#include <concepts>
#include <functional>
#include <string>

using namespace std;

// Hashable concept
template<typename T>
concept Hashable = requires(T x) {
    { hash<T>{}(x) } -> convertible_to<size_t>;
};

// HasLength concept: checks for .size() returning something convertible to size_t
template<typename T>
concept HasLength = requires(T x) {
    { x.size() } -> convertible_to<size_t>;
};

int main() {
    static_assert(HasLength<std::string>);
    static_assert(!HasLength<int>);
    static_assert(Hashable<int>);
    static_assert(Hashable<std::string>);

    cout << boolalpha;
    cout << "HasLength<string>: " << HasLength<std::string> << endl;
    cout << "HasLength<int>: " << HasLength<int> << endl;
    cout << "Hashable<int>: " << Hashable<int> << endl;
    cout << "Hashable<string>: " << Hashable<std::string> << endl;

    return 0;
}
