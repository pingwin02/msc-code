#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__device__ float math_function(float x) {
    return sinf(x) * sqrt(x);
}

__global__ void integrate_kernel(float *output, float start, float dx, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float x = start + i * dx;
        output[i] = math_function(x) * dx;
    }
}

extern "C" float launch_cuda_integration(float start, float end, int num_steps) {
    float *d_out;
    float *h_out;
    float total_area = 0;
    float dx = (end - start) / num_steps;
    size_t size = num_steps * sizeof(float);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (num_steps + threadsPerBlock - 1) / threadsPerBlock;

    h_out = (float*)malloc(size);
    cudaMalloc((void**)&d_out, size);
    integrate_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, start, dx, num_steps);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_steps; i++) {
        total_area += h_out[i];
    }

    cudaFree(d_out);
    free(h_out);

    return total_area;
}