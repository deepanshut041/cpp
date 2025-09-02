#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define CHECK(x) do { cudaError_t err=(x); if(err!=cudaSuccess){fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); return 1;}} while(0)


__global__ void addKernel(float *A, float* B, float *C, int n){
    auto j = threadIdx.x + blockDim.x * blockIdx.x;
    auto i = threadIdx.y + blockDim.y * blockIdx.y;

    auto ij = (i * n) + j;
    if (i < n && j < n)
    {
        C[ij] = A[ij] + B[ij];
    }
    
}

void vecAdd(int n){
    auto bytes = n * sizeof(float);

    auto A = (float**)malloc(n * sizeof(float*));
    auto B = (float**)malloc(n * sizeof(float*));
    auto C = (float**)malloc(n * sizeof(float*));
    
    for (int i = 0; i < n; i++) {
        A[i] = (float*)malloc(bytes);
        B[i] = (float*)malloc(bytes);
        C[i] = (float*)malloc(bytes);
    }

    
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i][j] = 1.0;
            B[i][j] = 2.0;
        }
        
    }
    

    auto cpu_start = chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    auto cpu_end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<milliseconds>(cpu_end - cpu_start);

    cout << "CPU Time: " << duration.count() << " milliseconds"<< endl;

    float *A_d, *B_d, *C_d;
    cudaEvent_t gpu_start, gpu_end;

    float gpu_duration;

    cudaMalloc((void**) &A_d, n * bytes);
    cudaMalloc((void**) &B_d, n * bytes);
    cudaMalloc((void**) &C_d, n * bytes);

    cudaMemcpy(A_d, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, bytes, cudaMemcpyHostToDevice);

    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);

    
    dim3 block(256, 256);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    cudaEventRecord(gpu_start);
    addKernel<<<grid, block>>>(A_d, B_d, C_d, n);
    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);

    cudaEventElapsedTime(&gpu_duration, gpu_start, gpu_end);


    cout << "GPU Time: " << gpu_duration  << " milliseconds"<< endl;

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    for (int i = 0; i < n; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }

    free(A);
    free(B);
    free(C);

}

int main(){
    vector<int> nums = {10, 100, 1000, 10000, 20000};

    for (const auto n: nums){
        cout << "Vector size: (" << n << " x " << n << ")" << endl;
        vecAdd(n);
        cout << endl;
    }


}