#include <iostream>
#include <vector>

using namespace std;

int minJumps(vector<int>& nums) {
    int l = nums.size();
    int ls = 0;
    int le = 0;
    int mJ = 0;

    while (le < l - 1) {
        int f = 0;
        for (int i = ls; i <= le; i++){
            f = max(f, i + nums[i]);
        }
        ls = le + 1;
        le = f;
        mJ++;
    }

    return mJ;
}

#ifndef TESTING
int main() {
    vector<int> nums1 = {2, 3, 1, 1, 4};
    cout << "Minimum jumps (Example 1): " << minJumps(nums1) << endl;

    vector<int> nums2 = {2, 3, 0, 1, 4};
    cout << "Minimum jumps (Example 2): " << minJumps(nums2) << endl;

    vector<int> nums3 = {0};
    cout << "Minimum jumps (Single Element): " << minJumps(nums3) << endl;

    vector<int> nums4 = {10, 1, 1, 1, 1};
    cout << "Minimum jumps (Large Jump): " << minJumps(nums4) << endl;

    vector<int> nums5 = {1, 1, 1, 1, 1};
    cout << "Minimum jumps (Multiple Jumps): " << minJumps(nums5) << endl;

    vector<int> nums6 = {1, 2, 3, 4, 5};
    cout << "Minimum jumps (Edge Case): " << minJumps(nums6) << endl;

    return 0;
}
#endif