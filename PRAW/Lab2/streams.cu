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
void histogram_kernel(int *data, int *histogram, int start_idx, int end_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start_idx;
    if (idx < end_idx) {
        atomicAdd(&histogram[data[idx]], 1);
    }
}

int main(int argc, char **argv) {
    int data_size = SIZE;
    if (argc > 1) {
        data_size = atoi(argv[1]);
    }

    int num_streams = 4;
    float milliseconds = 0;
    cudaEvent_t start, stop;

    int chunk_size = data_size / num_streams;
    int remainder = data_size % num_streams;

    int *h_data = NULL;
    int *h_histogram = (int*)malloc(RANGE * sizeof(int));

    if (cudaSuccess != cudaMallocHost((void **)&h_data, data_size * sizeof(int)))
        error_exit("Error allocating pinned memory");

    generate_data(h_data, data_size);

    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        if (cudaSuccess != cudaStreamCreate(&streams[i]))
            error_exit("Error creating stream");
    }

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

    int offset = 0;
    int threads_per_block = 1024;

    for (int i = 0; i < num_streams; i++) {
        int stream_chunk = chunk_size + (i < remainder ? 1 : 0);
        int blocks_in_grid = (stream_chunk + threads_per_block - 1) / threads_per_block;
        
        int start_idx = offset;
        int end_idx = start_idx + stream_chunk;

        cudaMemcpyAsync(&d_data[offset], &h_data[offset], stream_chunk * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
        histogram_kernel<<<blocks_in_grid, threads_per_block, 0, streams[i]>>>(d_data, d_histogram, start_idx, end_idx);

        offset += stream_chunk;
    }

    cudaDeviceSynchronize();

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

    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }

    free(h_histogram);
    cudaFreeHost(h_data);
    cudaFree(d_data);
    cudaFree(d_histogram);

    return 0;
}