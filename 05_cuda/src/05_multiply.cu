#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cassert>
#include <vector>

using namespace std;

#define SH_MEM_SIZE 16 * 16 * 4

void assertResult(int** A, int** B, int** C, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if (C[i][j] != B[i][j]){
                cerr << "Mismatch at (" << i << "," << j << "): "
                     << "expected " << B[i][j] << " but got " << C[i][j] << endl;
                assert(false && "Matrix multiplication result is incorrect!");
            }
        }
    }
    cout << "Result verified: C == B" << endl;
}

__global__ void mulKernel(const int* A, const int* B, int* C, int n){
    int r = threadIdx.y + blockIdx.y * blockDim.y;
    int c = threadIdx.x + blockIdx.x * blockDim.x;

    int t = 0;

    for (int i = 0; i < n; i++) {
        t += A[r * n + i] * B[i * n + c];
    }
    
    C[r * n + c] = t;
}

__global__ void tiledMulKernel(const int* A, const int* B, int* C, int n, int tile_size) {
    __shared__ int sA[SH_MEM_SIZE];
    __shared__ int sB[SH_MEM_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = ty + by * tile_size;
    int col = tx + bx * tile_size;

    int t = 0;

    for (int i = 0; i < (n + tile_size - 1) / tile_size; i++) {
        int tiled_col_A = i * tile_size + tx;
        int tiled_row_B = i * tile_size + ty;

        // Load tile from A
        if (row < n && tiled_col_A < n)
            sA[ty * tile_size + tx] = A[row * n + tiled_col_A];
        else
            sA[ty * tile_size + tx] = 0;

        // Load tile from B
        if (col < n && tiled_row_B < n)
            sB[ty * tile_size + tx] = B[tiled_row_B * n + col];
        else
            sB[ty * tile_size + tx] = 0;

        __syncthreads();

        // Multiply the loaded tiles
        for (int j = 0; j < tile_size; j++) {
            t += sA[ty * tile_size + j] * sB[j * tile_size + tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < n && col < n) {
        C[row * n + col] = t;
    }
}



void vecMul(int n){
    int rowBytes = n * sizeof(int);

    auto A = (int**)malloc(n * sizeof(int*));
    auto B = (int**)malloc(n * sizeof(int*));
    auto C = (int**)malloc(n * sizeof(int*));
    auto D = (int**)malloc(n * sizeof(int*));


    for (int i = 0; i < n; i++) {
        A[i] = (int*)malloc(rowBytes);
        B[i] = (int*)malloc(rowBytes);
        C[i] = (int*)malloc(rowBytes);
        D[i] = (int*)malloc(rowBytes);
    }


    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = i == j;
            B[i][j] = i + j;
        }
        
    }

    
    int *dA, *dB, *dD, *dC;
    cudaEvent_t start_gpu, end_gpu, start_gpu_tiled, end_gpu_tiles;

    cudaEventCreate(&start_gpu_tiled);
    cudaEventCreate(&end_gpu_tiles);
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);


    cudaMalloc((void**) &dA, n * rowBytes);
    cudaMalloc((void**) &dB, n * rowBytes);
    cudaMalloc((void**) &dC, n * rowBytes);
    cudaMalloc((void**) &dD, n * rowBytes);
    

    for (int i = 0; i < n; i++)
    {
        cudaMemcpy(dA + i * n, A[i], rowBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB + i * n, B[i], rowBytes, cudaMemcpyHostToDevice);
    }
    
    
    int tile_size = 16;
    dim3 blockDim(tile_size, tile_size);
    dim3 gridDim((n + tile_size - 1) / tile_size, (n + tile_size - 1) / tile_size);
    
    cudaEventRecord(start_gpu);
    mulKernel<<<gridDim, blockDim>>>(dA, dB, dC, n);
    cudaEventRecord(end_gpu);
    cudaEventSynchronize(end_gpu);

    cudaEventRecord(start_gpu_tiled);
    tiledMulKernel<<<gridDim, blockDim>>>(dA, dB, dD, n, tile_size);
    cudaEventRecord(end_gpu_tiles);
    cudaEventSynchronize(end_gpu_tiles);


    for (int i = 0; i < n; i++) cudaMemcpy(D[i], dD + i * n, rowBytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) cudaMemcpy(C[i], dC + i * n, rowBytes, cudaMemcpyDeviceToHost);
    
    float duration_gpu_tile = 0.0;
    float duration_gpu = 0.0;
    cudaEventElapsedTime(&duration_gpu, start_gpu, end_gpu);
    cudaEventElapsedTime(&duration_gpu_tile, start_gpu_tiled, end_gpu_tiles);
    assertResult(A, B, C, n);
    cout << "Time take by GPU: " << duration_gpu << " milliseconds" << endl;
    assertResult(A, B, D, n);
    cout << "Time take by GPU Tiled: " << duration_gpu_tile << " milliseconds" << endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);
    cudaFree(dC);

    for (int i = 0; i < n; i++)
    {
        free(A[i]);
        free(B[i]);
        free(C[i]);
        free(D[i]);
    }
    
    free(A);
    free(B);
    free(C);
    free(D);
}

int main(){
    vector<int> sizes = {10, 100, 500, 1000, 10000};
    for (auto size : sizes) {
        cout  << "Vector Size: (" << size << " x " << size << ")" << endl;
        vecMul(size);
        cout << endl;
    }
    
}