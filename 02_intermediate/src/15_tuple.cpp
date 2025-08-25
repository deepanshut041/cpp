#include <iostream>
#include <tuple>

using namespace std;

tuple<int, double> ReturnTuple(){
    return make_tuple(25, 15.2);
}

int main(){
    int i1, i2;
    double d1;

    tie(i1, d1) = ReturnTuple();
    cout << "i: " << i1 << " | d: " << d1 << endl;

    std::tie(i2, std::ignore) = ReturnTuple();
    cout << "i: " << i2 << endl;

    // Need cpp 17
    auto[i3, d3] = ReturnTuple();
    cout << "i: " << i3 << " | d: " << d3 << endl;

    // Avoid declarion
    auto t = ReturnTuple();
    cout << "i: " << get<0>(t) << " | d: " << get<1>(t) << endl;
}