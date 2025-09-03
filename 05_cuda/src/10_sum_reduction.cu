#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cuda.h>

using namespace std;

#define SH_SIZE 256
#define SH_MEM_SIZE (SH_SIZE)

__global__ void sumReduction(const float *input, float *output, int n){
    __shared__ float partial_sum[SH_MEM_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    partial_sum[threadIdx.x] = (tid < n) ? input[tid] : 0.0f;
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        if ((threadIdx.x % (2 * i)) == 0) {
            int j = threadIdx.x + i;
            if (j < blockDim.x) {
                partial_sum[threadIdx.x] += partial_sum[j];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[blockIdx.x] = partial_sum[0];
    }
}

void sum(int n){
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float *dA = nullptr, *dS = nullptr;

    size_t bytes = size_t(n) * sizeof(float);
    auto A = (float*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        A[i] = 1.0f;
    }

    auto start_cpu = chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        sum1 += A[i];
    }
    auto end_cpu = chrono::high_resolution_clock::now();

    auto duration_cpu = chrono::duration_cast<chrono::milliseconds>(end_cpu - start_cpu);

    cout << "Sum is: " << sum1 << endl;
    cout << "Cpu Time Taken: " << duration_cpu.count() << " Milliseconds" << endl;


    cudaMalloc((void**) &dA, bytes);
    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);

    int block = SH_SIZE;                        
    int grid  = (n + block - 1) / block;
    size_t partial_bytes = size_t(grid) * sizeof(float);
    cudaMalloc((void**) &dS, max(partial_bytes, sizeof(float)));

    cudaEvent_t start_gpu, end_gpu;
    float duration_gpu = 0.0f;

    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);

    cudaEventRecord(start_gpu);

    const float* pin  = dA;
    float*       pout = dS;
    int          cur_n = n;
    int          cur_grid = grid;

    sumReduction<<<cur_grid, block>>>(pin, pout, cur_n);
    cudaDeviceSynchronize();

    while (cur_grid > 1) {
        pin     = pout;                               
        cur_n   = cur_grid;
        cur_grid = (cur_n + block - 1) / block;
        pout = (pin == dS) ? dA : dS;

        sumReduction<<<cur_grid, block>>>(pin, pout, cur_n);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(end_gpu);
    cudaEventSynchronize(end_gpu);
    cudaEventElapsedTime(&duration_gpu, start_gpu, end_gpu);

    cudaMemcpy(&sum2, pout, sizeof(float), cudaMemcpyDeviceToHost);

    cout << "GPU Time Taken: " << duration_gpu << " Milliseconds" << endl;

    cudaFree(dA);
    cudaFree(dS);
    free(A);

    if (fabs(sum1 - sum2) > 1e-3f * max(1.0f, sum1)) {
        cerr << "Mismatch -> expected " << sum1 << " but got " << sum2 << endl;
        assert(false && "Reduction result is incorrect!");
    }
}

int main(){
    vector<int> nums = {1 << 16, 1 << 20, 1 << 24};

    for (auto &&n : nums) {
        cout << "Vector Size: " << n << endl;
        sum(n);
        cout << endl;
    }
}
