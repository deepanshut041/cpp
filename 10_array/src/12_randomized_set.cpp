#include <cstdlib>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <deque>
#include <random>

using namespace std;

class RandomizedSet {
    private:
        unordered_map<int, int> hashMap;
        deque<int> numbers;
    public:
        RandomizedSet() {
            numbers.clear();
            hashMap.clear();
        }

        bool insert(int val){
            if (hashMap.find(val) != hashMap.end()) return false;

            numbers.push_back(val);
            hashMap[val] = numbers.size() - 1;
            return true;
        }

        bool remove(int val){
            auto it = hashMap.find(val);
            if (it == hashMap.end()) return false;

            int idx = it->second;
            int lastVal = numbers.back();

            numbers[idx] = lastVal;
            hashMap[lastVal] = idx;

            numbers.pop_back();
            hashMap.erase(it);
            return true;
        }

        int getRandom(){
            int p = rand() % numbers.size();
            return numbers[p];
        }
};

int main(){

    RandomizedSet set;

    cout << (set.insert(1) ? "true" : "false") << endl;
    cout << (set.remove(2) ? "true" : "false") << endl;
    cout << (set.insert(2) ? "true" : "false") << endl;
    cout << set.getRandom() << endl;
    cout << (set.remove(1) ? "true" : "false") << endl;
    cout << (set.insert(2) ? "true" : "false") << endl;
    cout << set.getRandom() << endl;
}
