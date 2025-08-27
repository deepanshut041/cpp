#include <iostream>
#include <vector>

using namespace std;

int maxProfit(vector<int>& prices) {
    int mP = 0;
    int cP = 0;
    int min = prices[0];

    for (const int &p: prices){
        int nP = p - min;

        if(nP < cP){
            min = p;
            mP += cP;
            cP = 0;
        } else {
            cP = nP;
        }
    }
    return mP + cP;
}

int maxProfit2(vector<int>& prices) {
    int mP = 0;
    int s = prices[0];
    for (const int &p: prices){
        if (p > s){
            mP += p - s;
        }
        s = p;
    }
    return mP;
}

#ifndef TESTING
int main() {
    vector<int> prices1 = {7, 1, 5, 3, 6, 4};
    cout << "Max Profit (Example 1): " << maxProfit(prices1) << endl;
    cout << "Max Profit2 (Example 1): " << maxProfit2(prices1) << endl;

    vector<int> prices2 = {1, 2, 3, 4, 5};
    cout << "Max Profit (Example 2): " << maxProfit(prices2) << endl;
    cout << "Max Profit2 (Example 2): " << maxProfit2(prices2) << endl;

    vector<int> prices3 = {7, 6, 4, 3, 1};
    cout << "Max Profit (Example 3): " << maxProfit(prices3) << endl;
    cout << "Max Profit (Example 3): " << maxProfit2(prices3) << endl;

    return 0;
}
#endif