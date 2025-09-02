#include<iostream>
#include<vector>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace std::chrono;


__global__ void addKernel(const float* A, const float* B, float* C, int n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n){
        C[tid] = A[tid] + B[tid];
    }
}

void vec_add(int n){
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    A = (float*)malloc(n*sizeof(float));
    B = (float*)malloc(n*sizeof(float));
    C = (float*)malloc(n*sizeof(float));

    for(int i=0; i < n; i++){
        A[i] = i * 2.0;
        B[i] = i * 3.0;
    }

    auto start_cpu = high_resolution_clock::now();

    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }

    auto end_cpu = high_resolution_clock::now();
    auto duration_cpu = duration_cast<microseconds>(end_cpu - start_cpu);

    cudaEvent_t start_gpu, stop_gpu;

    cudaMalloc((void**) &d_A, n*sizeof(float));
    cudaMalloc((void**) &d_B, n*sizeof(float));
    cudaMalloc((void**) &d_C, n*sizeof(float));
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaMemcpy(d_A, A, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n*sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    
    cudaEventRecord(start_gpu);
    addKernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    
    float duration_gpu = 0;
    cudaEventElapsedTime(&duration_gpu, start_gpu, stop_gpu);


    cudaMemcpy(C, d_C, n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    cout << "Time taken by CPU: " << duration_cpu.count() / 1000.0 << " milliseconds" << endl;
    cout << "Time taken by GPU: " << duration_gpu << " milliseconds" << endl;


}

int main(){

    vector<int> a ={100, 1000, 100000, 1000000, 100000000};

    for(int i=0; i<a.size(); i++){
        cout << "Vector size: " << a[i] << endl;
        vec_add(a[i]);
        cout << endl;
    }
    return 0;
}