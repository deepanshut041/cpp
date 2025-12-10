#include <iostream>
using namespace std;

void log(int) { 
    cout << "Int\n"; 
}

template<typename T>
void log(T) { 
    cout << "General\n"; 
}

template<>
void log<double>(double) { 
    cout << "Double\n"; 
}

int main() {
    log(10);        // exact match → calls non-template int overload
    log(3.14);      // exact template specialization for double
    log(5.5f);      // float → no exact overload, so general
    log("hi");      // const char* → general

    return 0;
}
