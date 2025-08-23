#include <iostream>
#include <vector>

using namespace std;

void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
    int l = m + n - 1;
    m = m - 1;
    n = n - 1;

    while (n > -1){
        if (m > -1 && nums1.at(m) > nums2.at(n)){
            nums1[l] = nums1.at(m);
            m -= 1;
        } else {
            nums1[l] = nums2.at(n);
            n -= 1;
        }
        l -= 1;
    }
}

#ifndef TESTING
int main() {
    vector<int> nums1 = {1,2,3,0,0,0};
    vector<int> nums2 = {2,5,6};
    int m = 3, n = 3;

    cout << "Array num1: [" << ends;
    for (int i = 0; i < m; i++){
        cout << nums1.at(i) << (i < m - 1 ? ", " : "") << ends;
    }
    cout << "]" << endl;

    cout << "Array num2: [" << ends;
    for (int i = 0; i < n; i++){
        cout << nums2.at(i) << (i < n - 1 ? ", " : "") << ends;
    }
    cout << "]" << endl;

    merge(nums1, m ,nums2, n);

    cout << "Array num1: [" << ends;
    for (int i = 0; i < m + n; i++){
        cout << nums1.at(i) << (i < m + n - 1 ? ", " : "") << ends;
    }
    cout << "]" << endl;
}
#endif