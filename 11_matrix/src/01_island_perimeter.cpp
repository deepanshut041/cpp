#include <iostream>
#include <vector>

using namespace std;

int islandPerimeter(std::vector<std::vector<int>>& grid) {
    if (grid.empty() || grid[0].empty()) return 0;
    int r = grid.size(), c = grid[0].size(), p = 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            if (grid[i][j] == 0) continue;
            if (i == 0 || grid[i - 1][j] == 0) p++;
            if (i == r - 1 || grid[i + 1][j] == 0) p++;
            if (j == 0 || grid[i][j - 1] == 0) p++;
            if (j == c - 1 || grid[i][j + 1] == 0) p++;
        }
    }
    return p;
}

int main()
{

    vector<vector<int>> grid1 = {{0, 1, 0, 0}, {1, 1, 1, 0}, {0, 1, 0, 0}, {1, 1, 0, 0}};
    vector<vector<int>> grid2 = {{1}};
    vector<vector<int>> grid3 = {{1, 0}};

    int out1 = islandPerimeter(grid1);

    cout << "Output for [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]] -> " << out1 << endl;

    int out2 = islandPerimeter(grid2);
    cout << "Output for [[1]] -> " << out2 << endl;

    int out3 = islandPerimeter(grid3);
    cout << "Output for [[1,0]] -> " << out3 << endl;
}