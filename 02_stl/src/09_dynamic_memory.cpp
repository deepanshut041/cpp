#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main(){
    int *pointer_1 = new(nothrow) int;

    *pointer_1 = 6;

    cout << *pointer_1 << endl;

    delete pointer_1;
    pointer_1 = nullptr;

    if (!pointer_1){
        cout << "No mermory allocated";
    }

    cout << "pointer_1 deallocated\n";

    return 0;
}