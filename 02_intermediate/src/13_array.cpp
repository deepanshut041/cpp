#include <exception>
#include <iostream>
#include <array>

using namespace std;

void printScores(const array<int, 5> &scores, string name){
    cout << name << ": ";
    for (const auto& currentScore: scores)  // iterates by const reference
        cout << currentScore << ", ";
    cout << endl;
}

void printScores2(const array<int, 5> &scores, string name){
    cout << name << ": ";
    for (int i{0}; i < scores.size(); i++)  // iterates by const reference
        cout << scores[i] << ", ";
    cout << endl;
}

int main(){

    array<int, 5> intArr;
    intArr = {2, 3, 4, 5, 3};

    int intArr2[5] = {2, 3, 4, 5, 3};

    printScores(intArr, "Int Arr");
    printScores2(intArr, "Int Arr");

    try{
        cout << intArr.at(5) << endl;
    } catch (exception e){
        cerr << e.what() << "Handling custom error" << endl;
    }

    // This throws garbage value instead of error
    try{
        cout << intArr2[5] << endl;
    } catch (exception e){
        cerr << e.what() << "Handling custom error" << endl;
    }

}