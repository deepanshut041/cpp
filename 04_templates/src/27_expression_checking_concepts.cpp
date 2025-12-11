#include <iostream>
#include <vector>
#include <list>

using namespace std;

template <typename T>
concept HasPushBack = requires(T t) {
    t.push_back(typename T::value_type{});
};

void test(HasPushBack auto& c) {
    c.push_back(typename remove_reference_t<decltype(c)>::value_type{});
    cout << "push_back OK, size: " << c.size() << endl;
}

int main() {
    vector<int> v = {1, 2, 3};
    list<string> lst = {"a", "b"};

    test(v);
    test(lst);

    return 0;
}
