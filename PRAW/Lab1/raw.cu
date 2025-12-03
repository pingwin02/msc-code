#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void computeSumRaw(int *data, unsigned long long *globalSum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(globalSum, (unsigned long long)data[idx]);
    }
}

void generateRandomNumbers(int *arr, int N, int A, int B) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        arr[i] = A + rand() % (B - A + 1);
    }
}

int main(int argc, char **argv) {
    int threadsPerBlock = 1024;
    int blocksInGrid;
    int N;
    int A = 0; 
    int B = 100;
    
    if (argc >= 2) {
        N = atoi(argv[1]);
    } else {
        printf("Enter number of elements: ");
        scanf("%d", &N);
    }

    int *randomNumbers = (int *)malloc(N * sizeof(int));
    if (randomNumbers == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    generateRandomNumbers(randomNumbers, N, A, B);

    int *d_randomNumbers;
    unsigned long long *d_globalSum;
    
    cudaMalloc((void **)&d_randomNumbers, N * sizeof(int));
    cudaMalloc((void **)&d_globalSum, sizeof(unsigned long long));

    cudaMemcpy(d_randomNumbers, randomNumbers, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_globalSum, 0, sizeof(unsigned long long));

    blocksInGrid = ceil((double)N / threadsPerBlock);
    
    printf("RAW: Running with %d blocks, %d threads/block\n", blocksInGrid, threadsPerBlock);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    cudaEventRecord(start, 0);

    computeSumRaw<<<blocksInGrid, threadsPerBlock>>>(d_randomNumbers, d_globalSum, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    unsigned long long h_sum = 0;
    cudaMemcpy(&h_sum, d_globalSum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    double average = (double)h_sum / N;

    printf("RAW Result:\n");
    printf("  Total Sum: %llu\n", h_sum);
    printf("  Average:   %f\n", average);
    printf("  Time:      %.3f ms\n", milliseconds);

    free(randomNumbers);
    cudaFree(d_randomNumbers);
    cudaFree(d_globalSum);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}