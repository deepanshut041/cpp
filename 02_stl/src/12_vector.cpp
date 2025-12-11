#include <iostream>
#include <vector>

using namespace std;


void printScores(vector<int> &scores, string name){
    cout << name << ": ";
    for (const auto& currentScore: scores)  // iterates by const reference
        cout << currentScore << ", ";
    cout << endl;
}

int main(){
    vector<int> total_scores(5, 5);
    vector<int> scores_1 = {1, 2, 3, 4, 5};
    vector<int> scores_2 {1, 3, 5, 7, 8};

    printScores(scores_1, "Score 1");
    printScores(scores_1, "Score 2");
    printScores(total_scores, "Total Score Before");

    for(int i=0; i < scores_1.size(); i ++){
        total_scores[i] = scores_1[i] + scores_2[i];
    }
    printScores(total_scores, "Total Score");

    total_scores.resize(3);
    printScores(total_scores, "Resize Score");

    total_scores.resize(5);
    printScores(total_scores, "Resize Score");

}