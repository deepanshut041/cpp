#include <iostream>
#include <list>

using namespace std;

void printScores(list<int> &arr){
    cout << "Arr: ";
    for (const auto& a: arr)  // iterates by const reference
        cout << a << ", ";
    cout << endl;
}

int main(){
    list<int> l = {1, 2, 3, 4};
    printScores(l);

    l.push_front(0);
    printScores(l);

    l.push_back(5);
    printScores(l);

    l.pop_back();
    printScores(l);

    l.pop_front();
    printScores(l);
}