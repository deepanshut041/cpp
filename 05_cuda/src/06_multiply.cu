#include <iostream>
#include <cuda.h>
#include <vector>
#include <chrono>
#include <cassert>

using namespace std;


__global__ void mulKernel(int *A, int *B, int *C, int M, int N, int K){
    auto r = threadIdx.y + blockDim.y * blockIdx.y;
    auto c = threadIdx.x + blockDim.x * blockIdx.x;
    

    if (r < M && c < K) {
        int t = 0;
        for (int p = 0; p < N; p++) {
            t += A[r * N + p] * B[p * K + c];
        }
        
        C[r * K + c] = t;
    }
    
}

void assertResult(int** A, int** B, int m, int k){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < k; j++){
            if (A[i][j] != B[i][j]){
                cerr << "Mismatch at (" << i << "," << j << "): "
                     << "expected " << A[i][j] << " but got " << B[i][j] << endl;
                assert(false && "Matrix multiplication result is incorrect!");
            }
        }
    }
    cout << "Result verified!" << endl;
}

void vecMul(int m, int n, int k){

    auto A = (int**)malloc(m * sizeof(int*));
    auto B = (int**)malloc(n * sizeof(int*));
    auto C = (int**)malloc(m * sizeof(int*));
    auto D = (int**)malloc(m * sizeof(int*));

    for (auto i = 0; i < n; i++)
    {
        
        B[i] = (int*)malloc(n * sizeof(int));   
    }

    for (auto i = 0; i < m; i++)
    {
        
        A[i] = (int*)malloc(n * sizeof(int));
        C[i] = (int*)malloc(k * sizeof(int));
        D[i] = (int*)malloc(k * sizeof(int));;   
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
    for (auto i = 0; i < m; i++)
    {
        for (auto j = 0; j < k; j++)
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

    cout << "CPU Time: " << duration_cpu.count() << " milliseconds"<< endl;

    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);

    int *dA, *dB, *dD;
    dim3 block(16, 16);
    dim3 grid((k + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    cudaMalloc((void**) &dA, m * n * sizeof(int));
    cudaMalloc((void**) &dB, n * k * sizeof(int));
    cudaMalloc((void**) &dD, m * k * sizeof(int));

    for (int i = 0; i < m; ++i) cudaMemcpy(dA + i * n, A[i], n * sizeof(int), cudaMemcpyHostToDevice);
    for (int i = 0; i < n; ++i) cudaMemcpy(dB + i * k, B[i], k * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start_gpu);
    mulKernel<<<grid, block>>>(dA, dB, dD, m, n, k);
    cudaEventRecord(end_gpu);
    cudaEventSynchronize(end_gpu);

    for (int i = 0; i < m; ++i) cudaMemcpy(D[i], dD + i * k, k * sizeof(int), cudaMemcpyDeviceToHost);
    assertResult(C, D, m, k);

    float duration_gpu;

    cudaEventElapsedTime(&duration_gpu, start_gpu, end_gpu);


    cout << "GPU Time: " << duration_gpu  << " milliseconds"<< endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);

    for (int i = 0; i < n; i++) {
        free(B[i]);
    }

    for (int i = 0; i < m; i++) {
        free(A[i]);
        free(C[i]);
        free(D[i]);
    }

    free(A);
    free(B);
    free(C);
    free(D);
}

int main(){
    vector<int> nums{32, 64, 128, 256, 512, 1024};
    for (auto i : nums)
    {
        cout << "Vector Size: (" << i << " x " << i << ")" << endl; 
        vecMul(i, i / 2, i / 4);
        cout << endl;
    }
    
}