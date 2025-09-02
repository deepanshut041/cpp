#include<iostream>
#include<vector>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace std::chrono;


__global__ void addKernel(const int *A, const int *B, int *C, int n){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < n)
    {
        C[tid] = A[tid] + B[tid];
    }
    
}

void vec_add(int n){
    // int id = cudaGetDevice(&id);
    int *A, *B, *C;

    cudaMallocManaged(&A, n * sizeof(int));
    cudaMallocManaged(&B, n * sizeof(int));
    cudaMallocManaged(&C, n * sizeof(int));

    for (int i = 0; i < n; i++) {
        A[i] = i;
        B[i] = i;
    }

    auto start_clock = high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
    auto end_clock = high_resolution_clock::now();
    auto duration_cpu = duration_cast<milliseconds>(end_clock - start_clock);
    
    int threads = 256;
    int grid = (n + 256 - 1) / 256;

    // cudaMemPrefetchAsync(A, n * sizeof(int), id);
    // cudaMemPrefetchAsync(B, n * sizeof(int), id);

    cudaEvent_t cuda_start, cuda_end;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_end);

    cudaEventRecord(cuda_start);
    addKernel<<<grid, threads>>>(A, B, C, n);
    cudaEventRecord(cuda_end);
    cudaEventSynchronize(cuda_end);

    // cudaMemPrefetchAsync(B, n * sizeof(int), cudaCpuDeviceId);

    float duration_gpu = 0;
    cudaEventElapsedTime(&duration_gpu, cuda_start, cuda_end);
    
    cout << "Cpu time: " << duration_cpu.count() << "Ms" << endl;
    cout << "GPU time: " << (duration_gpu) << "Ms" << endl;


    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    
}   


int main(){

    vector<int> a ={100000, 1000000, 10000000, 100000000};

    for(int i=0; i<a.size(); i++){
        cout << "Vector size: " << a[i] << endl;
        vec_add(a[i]);
        cout << endl;
    }
    return 0;
}