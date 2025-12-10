#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define MAX_DEPTH 16
#define THRESHOLD 32

__host__ void errorexit(const char *s) {
    printf("\n%s\n", s);
    exit(EXIT_FAILURE);
}

__device__ void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

__device__ void selection_sort(int *data, int left, int right) {
    for (int i = left; i <= right; ++i) {
        int min_idx = i;
        for (int j = i + 1; j <= right; ++j) {
            if (data[j] < data[min_idx]) {
                min_idx = j;
            }
        }
        swap(&data[i], &data[min_idx]);
    }
}

__global__ void quicksort_kernel(int *data, int left, int right, int depth) {
    if (left >= right) return;

    if (depth >= MAX_DEPTH || (right - left) < THRESHOLD) {
        selection_sort(data, left, right);
        return;
    }

    int pivot = data[right];
    int i = (left - 1);

    for (int j = left; j <= right - 1; j++) {
        if (data[j] <= pivot) {
            i++;
            swap(&data[i], &data[j]);
        }
    }
    swap(&data[i + 1], &data[right]);
    int pi = i + 1;

    if (threadIdx.x == 0) {
        cudaStream_t s_left, s_right;

        cudaStreamCreateWithFlags(&s_left, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&s_right, cudaStreamNonBlocking);

        if (left < pi - 1) {
            quicksort_kernel<<<1, 1, 0, s_left>>>(data, left, pi - 1, depth + 1);
        }

        if (pi + 1 < right) {
            quicksort_kernel<<<1, 1, 0, s_right>>>(data, pi + 1, right, depth + 1);
        }

        cudaStreamDestroy(s_left);
        cudaStreamDestroy(s_right);
    }
}

int main(int argc, char **argv) {
    int n = 10000;
    if (argc > 1) n = atoi(argv[1]);

    size_t size = n * sizeof(int);
    int *h_data = (int *)malloc(size);
    int *d_data;

    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        h_data[i] = rand() % n;
    }

    if (cudaSuccess != cudaMalloc((void **)&d_data, size))
        errorexit("Error allocating memory on GPU");

    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    if (cudaSuccess != cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH))
        errorexit("Error setting sync depth limit");

    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32768);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    quicksort_kernel<<<1, 1>>>(d_data, 0, n - 1, 0);

    if (cudaSuccess != cudaGetLastError())
        errorexit("Error during kernel launch");

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    int sorted = 1;
    for (int i = 0; i < n - 1; i++) {
        if (h_data[i] > h_data[i + 1]) {
            sorted = 0;
            break;
        }
    }

    if (sorted) {
        printf("SUCCESS: Array sorted correctly.\n");
    } else {
        printf("FAILURE: Array is not sorted.\n");
    }
    printf("Execution time: %.3f ms\n", milliseconds);

    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}