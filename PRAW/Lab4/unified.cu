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
    int *result;
    float milliseconds = 0;
    cudaEvent_t start, stop;

    if (cudaSuccess != cudaMallocManaged(&result, sizeof(int)))
        errorexit("Error allocating unified memory");

    *result = 1;
    if (target_number < 2) *result = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (*result) {
        int threadsinblock = 1024;
        int blocksingrid = 128; 

        cudaEventRecord(start);

        check_prime_kernel<<<blocksingrid, threadsinblock>>>(target_number, result);

        cudaEventRecord(stop);

        if (cudaSuccess != cudaGetLastError())
            errorexit("Error during kernel launch");
    
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
    }

    if (*result)
        printf("%lld is a prime number.\n", target_number);
    else
        printf("%lld is NOT a prime number.\n", target_number);

    printf("Execution time: %.3f ms\n", milliseconds);

    if (cudaSuccess != cudaFree(result))
        errorexit("Error freeing memory");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
