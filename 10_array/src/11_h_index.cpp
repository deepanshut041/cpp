#include <iostream>
#include <vector>

using namespace std;

int hIndex(vector<int>& citations) {
    int N = citations.size();
    int h = 0;
    int r = 0;
    vector<int> p_counts(N + 1, 0);

    for (const auto c: citations){
        if (c >= N){
            p_counts[N] += 1;
        } else {
            p_counts[c] += 1;
        }
    }

    for (int i = N; i > -1; i--){
        h += p_counts[i];
        if (h >= i){
            r = i;
            break;
        }
    }

    return r;
}


int main(){
    vector<int> citations1{3,0,6,1,5};
    int r1 = hIndex(citations1);
    cout << "Citations: [3,0,6,1,5]" << "Output: " << r1 << endl;


    vector<int> citations2{1,3,1};
    int r2 = hIndex(citations2);
    cout << "Citations: [1,3,1]" << "Output: " << r2 << endl;
}