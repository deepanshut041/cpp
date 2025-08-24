#include <iostream>
#include <vector>

using namespace std;

int removeDuplicates(vector<int>& nums) {
    int l = nums.size();
    if ( l <= 2) return l;
    int s = 2;

    for (int i = 2; i < l; i++){
        if (nums[i] != nums[s-2]){
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