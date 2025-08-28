#include <iostream>
#include <vector>

using namespace std;

bool canJump(vector<int>& nums) {
    int l = nums.size() - 1;

    for(int i = l; i > -1; i--){
        if (i + nums[i] >= l){
            l =  i;
        }
    }

    return l == 0;
}

#ifndef TESTING
int main() {
    vector<int> nums1 = {2, 3, 1, 1, 4};
    cout << "Can jump (Example 1): " << (canJump(nums1) ? "true" : "false") << endl;

    vector<int> nums2 = {3, 2, 1, 0, 4};
    cout << "Can jump (Example 2): " << (canJump(nums2) ? "true" : "false") << endl;

    vector<int> nums3 = {0};
    cout << "Can jump (Single Element): " << (canJump(nums3) ? "true" : "false") << endl;

    vector<int> nums4 = {0, 0, 0, 0};
    cout << "Can jump (All Zeros): " << (canJump(nums4) ? "true" : "false") << endl;

    vector<int> nums5 = {10, 1, 1, 1, 1};
    cout << "Can jump (Large Jump): " << (canJump(nums5) ? "true" : "false") << endl;

    vector<int> nums6 = {1, 0, 0, 0, 0};
    cout << "Can jump (Edge Case): " << (canJump(nums6) ? "true" : "false") << endl;

    return 0;
}
#endif