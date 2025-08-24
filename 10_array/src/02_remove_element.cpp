#include <iostream>
#include <vector>

using namespace std;

int removeElement(vector<int>& nums, int val){
    int s = 0;

    for (int e = 0; e < nums.size(); ++e) {
        if (nums[e] != val) {
            nums[s] = nums[e];
            s++;
        }
    }

    return s;
}

#ifndef TESTING
int main() {
    vector<int> nums = {0, 1, 2, 2, 3, 0, 4, 2};
    int val = 2;

    // Print the array before calling removeElement
    cout << "Array before: ";
    for (int num : nums) {
        cout << num << " ";
    }
    cout << endl;

    // Call removeElement
    int k = removeElement(nums, val);

    // Print the array after calling removeElement
    cout << "Array after: ";
    for (int i = 0; i < k; ++i) {
        cout << nums[i] << " ";
    }
    cout << endl;

    // Print the number of elements not equal to val
    cout << "Number of elements not equal to " << val << ": " << k << endl;

    return 0;
}
#endif