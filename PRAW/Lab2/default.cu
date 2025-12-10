#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define RANGE 1000
#define SIZE 10000000

__host__
void error_exit(const char *s) {
    printf("\n%s\n", s);
    exit(EXIT_FAILURE);
}

__host__
void generate_data(int *data, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        data[i] = rand() % RANGE;
    }
}

__global__
void histogram_kernel(int *data, int *histogram, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&histogram[data[idx]], 1);
    }
}

int main(int argc, char **argv) {
    int data_size = SIZE;
    if (argc > 1) {
        data_size = atoi(argv[1]);
    }

    int *h_data = (int*)malloc(data_size * sizeof(int));
    int *h_histogram = (int*)malloc(RANGE * sizeof(int));
    float milliseconds = 0;
    cudaEvent_t start, stop;

    generate_data(h_data, data_size);

    int *d_data = NULL;
    int *d_histogram = NULL;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    if (cudaSuccess != cudaMalloc((void **)&d_data, data_size * sizeof(int)))
        error_exit("Error allocating memory");

    if (cudaSuccess != cudaMalloc((void **)&d_histogram, RANGE * sizeof(int)))
        error_exit("Error allocating memory");

    if (cudaSuccess != cudaMemset(d_histogram, 0, RANGE * sizeof(int)))
        error_exit("Error initializing memory");

    if (cudaSuccess != cudaMemcpy(d_data, h_data, data_size * sizeof(int), cudaMemcpyHostToDevice))
        error_exit("Error copying data");

    int threads_per_block = 1024;
    int blocks_per_grid = (data_size + threads_per_block - 1) / threads_per_block;

    histogram_kernel<<<blocks_per_grid, threads_per_block>>>(d_data, d_histogram, data_size);

    if (cudaSuccess != cudaMemcpy(h_histogram, d_histogram, RANGE * sizeof(int), cudaMemcpyDeviceToHost))
        error_exit("Error copying results");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    long total_count = 0;
    for (int i = 0; i < RANGE; i++) {
        total_count += h_histogram[i];
    }

    printf("Total count: %ld\n", total_count);
    printf("Execution time: %.3f ms\n", milliseconds);

    free(h_data);
    free(h_histogram);
    cudaFree(d_data);
    cudaFree(d_histogram);

    return 0;
}