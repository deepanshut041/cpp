#include <iostream>
#include <vector>

using namespace std;


int maxProfit(vector<int>& prices) {
    int maxProfit = 0;
    int minStock = prices[0];

    for(const int &p: prices){
        minStock = p < minStock ? p: minStock;
        maxProfit = (p - minStock) > maxProfit ? (p - minStock): maxProfit;
    }

    return maxProfit;
}

#ifndef TESTING
#include <iostream>
using namespace std;

int main() {
    vector<int> prices1 = {7, 1, 5, 3, 6, 4};
    cout << "Max Profit: " << maxProfit(prices1) << endl;

    vector<int> prices2 = {7, 6, 4, 3, 1};
    cout << "Max Profit: " << maxProfit(prices2) << endl;

    return 0;
}
#endif