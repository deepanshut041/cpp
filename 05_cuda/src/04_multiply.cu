#include <iostream>
#include <cuda.h>
#include <vector>
#include <chrono>
#include <cassert>

using namespace std;

__global__ void mulKernel(int *A, int *B, int *C, int n){
    auto j = threadIdx.x + blockDim.x * blockIdx.x;
    auto i = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < n && j < n)
    {
        int t = 0;
        for (int k = 0; k < n; k++)
        {
            t += A[i * n + k] * B[k * n + j];
        }
        
        C[i * n + j] = t;
    }
    
}

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

void vecMul(int n){
    int bytes = n * sizeof(int);

    auto A = (int**)malloc(n * sizeof(int*));
    auto B = (int**)malloc(n * sizeof(int*));
    auto C = (int**)malloc(n * sizeof(int*));
    auto D = (int**)malloc(n * sizeof(int*));

    for (auto i = 0; i < n; i++)
    {
        A[i] = (int*)malloc(bytes);
        B[i] = (int*)malloc(bytes);
        C[i] = (int*)malloc(bytes);
        D[i] = (int*)malloc(bytes);   
    }

    for (auto i = 0; i < n; i++)
    {
        for (auto j = 0; j < n; j++)
        {
            A[i][j] = i == j ? 1: 0;
            B[i][j] = 1;
        }
    }

    auto start_cpu = chrono::high_resolution_clock::now();
    for (auto i = 0; i < n; i++)
    {
        for (auto j = 0; j < n; j++)
        {
            int p = 0;

            for (int k = 0; k < n; k++)
            {
                p += A[i][k] * B[k][j];
            }
            
            C[i][j] = p;
            
        }
    }
    auto end_cpu = chrono::high_resolution_clock::now();
    auto duration_cpu = chrono::duration_cast<chrono::milliseconds>(end_cpu - start_cpu);

    assertResult(A, B, C, n);
    cout << "CPU Time: " << duration_cpu.count() << " milliseconds"<< endl;

    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);

    int *dA, *dB, *dC;
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    cudaMalloc((void**) &dA, n * bytes);
    cudaMalloc((void**) &dB, n * bytes);
    cudaMalloc((void**) &dC, n * bytes);

    for (int i = 0; i < n; ++i) cudaMemcpy(dA + i*n, A[i], bytes, cudaMemcpyHostToDevice);
    for (int i = 0; i < n; ++i) cudaMemcpy(dB + i*n, B[i], bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(start_gpu);
    mulKernel<<<grid, block>>>(dA, dB, dC, n);
    cudaEventRecord(end_gpu);
    cudaEventSynchronize(end_gpu);

    for (int i = 0; i < n; ++i) cudaMemcpy(D[i], dC + i*n, bytes, cudaMemcpyDeviceToHost);
    assertResult(A, B, D, n);

    float duration_gpu;

    cudaEventElapsedTime(&duration_gpu, start_gpu, end_gpu);


    cout << "GPU Time: " << duration_gpu  << " milliseconds"<< endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    for (int i = 0; i < n; i++) {
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
    vector<int> nums{10, 100, 1000};
    for (auto i : nums)
    {
        cout << "Vector Size: (" << i << " x " << i << ")" << endl; 
        vecMul(i);
        cout << endl;
    }
    
}