#include <iostream>
#include <utility>
#include <vector>

using namespace std;

void rotate(vector<int>& nums, int k){
    int l = nums.size();
    k = k % l;
    if (k == 0 || l <= 1) return;

    for (int i = 0; i < l / 2; ++i) {
        swap(nums[i], nums[l - 1 - i]);
    }

    for (int i = 0; i < k / 2; ++i) {
        swap(nums[i], nums[k - 1 - i]);
    }

    for (int i = 0; i < (l - k) / 2; ++i) {
        swap(nums[k + i], nums[l - 1 - i]);
    }
}

#ifndef TESTING
int main() {
    vector<int> nums1 = {1, 2, 3, 4, 5, 6, 7};
    int k1 = 3;
    rotate(nums1, k1);
    cout << "Rotated Array (k=3): ";
    for (int num : nums1) {
        cout << num << " ";
    }
    cout << endl;

    vector<int> nums2 = {-1, -100, 3, 99};
    int k2 = 2;
    rotate(nums2, k2);
    cout << "Rotated Array (k=2): ";
    for (int num : nums2) {
        cout << num << " ";
    }
    cout << endl;

    vector<int> nums3 = {1, 2, 3, 4};
    int k3 = 0;
    rotate(nums3, k3);
    cout << "Rotated Array (k=0): ";
    for (int num : nums3) {
        cout << num << " ";
    }
    cout << endl;

    vector<int> nums4 = {1, 2, 3, 4};
    int k4 = 4;
    rotate(nums4, k4);
    cout << "Rotated Array (k=4): ";
    for (int num : nums4) {
        cout << num << " ";
    }
    cout << endl;

    vector<int> nums5 = {1, 2, 3, 4, 5};
    int k5 = 7; // Equivalent to k=2
    rotate(nums5, k5);
    cout << "Rotated Array (k=7): ";
    for (int num : nums5) {
        cout << num << " ";
    }
    cout << endl;

    vector<int> nums6 = {1};
    int k6 = 3;
    rotate(nums6, k6);
    cout << "Rotated Array (k=3, single element): ";
    for (int num : nums6) {
        cout << num << " ";
    }
    cout << endl;

    return 0;
}
#endif