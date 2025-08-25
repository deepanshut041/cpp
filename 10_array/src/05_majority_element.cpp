#include <iostream>
#include <vector>

using namespace std;

int majorityElement(const vector<int>& nums){
    int m = nums[0];
    int c = 1;

    for(const auto &n: nums){
        if(n == m){
            c++;
        } else {
            c--;
            if (c == 0){
                m = n;
                c++;
            }
        }
    }

    return m;
}

#ifndef TESTING
int main() {
    vector<int> nums1 = {3, 2, 3};
    cout << "Majority Element in {3, 2, 3}: " << majorityElement(nums1) << endl;

    vector<int> nums2 = {2, 2, 1, 1, 1, 2, 2};
    cout << "Majority Element in {2, 2, 1, 1, 1, 2, 2}: " << majorityElement(nums2) << endl;

    vector<int> nums3 = {1};
    cout << "Majority Element in {1}: " << majorityElement(nums3) << endl;

    vector<int> nums4 = {5, 5, 5, 5};
    cout << "Majority Element in {5, 5, 5, 5}: " << majorityElement(nums4) << endl;

    vector<int> nums5(1000, 1);
    nums5.insert(nums5.end(), 500, 2);
    cout << "Majority Element in large input: " << majorityElement(nums5) << endl;

    vector<int> nums6 = {-1, -1, -1, 2, 2};
    cout << "Majority Element in {-1, -1, -1, 2, 2}: " << majorityElement(nums6) << endl;

    vector<int> nums7 = {1, 1, 2, 2, 2, 1, 1, 1};
    cout << "Majority Element in {1, 1, 2, 2, 2, 1, 1, 1}: " << majorityElement(nums7) << endl;

    return 0;
}
#endif