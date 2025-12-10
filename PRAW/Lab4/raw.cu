#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s", s); 
    exit(EXIT_FAILURE);   
}

__global__ 
void check_prime_kernel(long long number, int *is_prime) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x + 2;

    while (idx * idx <= number) { 
        if (number % idx == 0) {
            *is_prime = 0;
            return; 
        }
        idx += blockDim.x * gridDim.x;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) errorexit("Usage: ./program <number>");

    long long target_number = atoll(argv[1]);
    int h_result = 1; 
    int *d_result = NULL;
    float milliseconds = 0;
    cudaEvent_t start, stop;

    if (target_number < 2) h_result = 0;
    
    if (h_result) {
        if (cudaSuccess != cudaMalloc((void **)&d_result, sizeof(int)))
            errorexit("Error allocating memory");

        if (cudaSuccess != cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice))
            errorexit("Error copying to device");

        int threadsinblock = 1024;
        int blocksingrid = 128; 

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        check_prime_kernel<<<blocksingrid, threadsinblock>>>(target_number, d_result);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        if (cudaSuccess != cudaGetLastError())
            errorexit("Error during kernel launch");

        if (cudaSuccess != cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost))
            errorexit("Error copying results");
        
        if (cudaSuccess != cudaFree(d_result))
            errorexit("Error freeing memory");
    }

    if (h_result)
        printf("%lld is a prime number.\n", target_number);
    else
        printf("%lld is NOT a prime number.\n", target_number);

        printf("Execution time: %.3f ms\n", milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

    return 0;
}