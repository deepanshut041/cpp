#include <iostream>

using namespace std;

// For each wornt wrok here
//int printScores(int scores[]){
//    cout << "Scores: ";
//    for (const auto& currentScore: scores)  // iterates by const reference
//        cout << currentScore << ", ";
//    cout << endl;
//}

int main (){
    // Create a fixed-size array of 7 integers with initial values
    int scores[7] = {1, 2, 5, 6, 7, 8, 9};

    // First loop: using range-based for with 'const auto'
    // - Each element of 'scores' is copied into currentScore (read-only copy)
    // - Good for simple values, but may cause unnecessary copies for large objects
    cout << "Scores: ";
    for (const auto currentScore: scores)   // iterates by value (copy)
        cout << currentScore << ", ";
    cout << endl;

    // Second loop: using range-based for with 'const auto&'
    // - Each element of 'scores' is accessed by reference (read-only reference)
    // - Avoids copies, more efficient especially for complex objects
    // - For primitive types like int, there's no real performance difference
    cout << "Scores: ";
    for (const auto& currentScore: scores)  // iterates by const reference
        cout << currentScore << ", ";
    cout << endl;
}