#include <iostream>
#include <vector>

using namespace std;

int removeDuplicates(vector<int>& nums) {

    if (nums.size() == 0) return 0;
    int s = 1;

    for (int i = 1; i < nums.size(); i++){
        if (nums.at(i) != nums.at(i-1)){
            nums[s] = nums[i];
            s++;
        }
    }

    return s;
}

#ifndef TESTING
int main() {
    vector<int> nums = {0, 0, 1, 1, 1, 2, 2, 3, 3, 4};

    int k = removeDuplicates(nums);

    cout << "Number of unique elements: " << k << endl;
    cout << "Modified array: ";
    for (int i = 0; i < k; ++i) {
        cout << nums[i] << " ";
    }
    cout << endl;

    return 0;
}
#endif