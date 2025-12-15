#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TILE_SIZE 16
#define N_STREAMS 8
#define BLOCK_SIZE 1024

typedef void (*matmul_func)(double*, double*, double*, double*, double*, double*, int);

void init_matrix(double** M, int N) {
    cudaMallocHost((void**)M, N * N * sizeof(double));
    for (int i = 0; i < N * N; i++) {
        (*M)[i] = (double)rand() / RAND_MAX;
    }
}

void matmul_serial(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void matmul_kernel_trivial(double* A, double* B, double* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N * N) {
        int row = idx % N;
        int col = idx / N;

        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_kernel_optimized(double* A, double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launch_cuda_trivial(double* h_A, double* h_B, double* h_C, double* d_A, double* d_B, double* d_C, int N) {
    size_t size = N * N * sizeof(double);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int numBlocks = (N * N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    matmul_kernel_trivial<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
}

void launch_cuda_optimized(double* h_A, double* h_B, double* h_C, double* d_A, double* d_B, double* d_C, int N) {
    size_t size = N * N * sizeof(double);
    int rows_per_stream = N / N_STREAMS;
    size_t chunk_size = rows_per_stream * N * sizeof(double);

    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, 0);

    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) cudaStreamCreate(&streams[i]);

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (rows_per_stream + TILE_SIZE - 1) / TILE_SIZE);

    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * rows_per_stream * N;
        
        cudaMemcpyAsync(d_A + offset, h_A + offset, chunk_size, cudaMemcpyHostToDevice, streams[i]);
        matmul_kernel_optimized<<<dimGrid, dimBlock, 0, streams[i]>>>(d_A + offset, d_B, d_C + offset, N);
        cudaMemcpyAsync(h_C + offset, d_C + offset, chunk_size, cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
}

void benchmark(matmul_func func, double* h_A, double* h_B, double* h_C, double* d_A, double* d_B, double* d_C, int N, const char* name, double* avg_time, double* uncertainty, int num_runs) {
    double sum = 0;
    double min_time = 1e9;
    double max_time = 0;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Benchmarking %s (%d runs)...\n", name, num_runs); fflush(stdout);

    func(h_A, h_B, h_C, d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < num_runs; i++) {
        float milliseconds = 0;
        
        cudaEventRecord(start, 0);
        func(h_A, h_B, h_C, d_A, d_B, d_C, N);
        cudaEventRecord(stop, 0);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        double time_sec = milliseconds / 1000.0;
        
        sum += time_sec;
        if (time_sec < min_time) min_time = time_sec;
        if (time_sec > max_time) max_time = time_sec;
    }

    *avg_time = sum / num_runs;
    *uncertainty = (max_time - min_time) / 2.0;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int verify(double* C1, double* C2, int N) {
    double epsilon = 1e-5;
    for (int i = 0; i < N * N; i++) {
        if (fabs(C1[i] - C2[i]) > epsilon) {
            fprintf(stderr, "Verification FAILED at index %d: %.10f vs %.10f\n", i, C1[i], C2[i]);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int num_runs = 10;
    
    if (N % N_STREAMS != 0) {
        printf("Error: N must be divisible by %d\n", N_STREAMS);
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("=== Matrix Multiplication CUDA %dx%d | Device: %s ===\n", N, N, prop.name);

    double *h_A, *h_B, *h_C_serial, *h_C_gpu_result;
    init_matrix(&h_A, N);
    init_matrix(&h_B, N);
    cudaMallocHost((void**)&h_C_gpu_result, N * N * sizeof(double));
    h_C_serial = (double*)calloc(N * N, sizeof(double));

    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(double));
    cudaMalloc((void**)&d_B, N * N * sizeof(double));
    cudaMalloc((void**)&d_C, N * N * sizeof(double));

    double t_serial_avg = 0.0;
    double t_triv_avg, t_triv_unc;
    double t_opt_avg, t_opt_unc;

    printf("Serial version (baseline)...\n");
    clock_t s = clock();
    matmul_serial(h_A, h_B, h_C_serial, N);
    clock_t e = clock();
    t_serial_avg = (double)(e - s) / CLOCKS_PER_SEC;

    cudaMemset(d_C, 0, N * N * sizeof(double));
    benchmark(launch_cuda_trivial, h_A, h_B, h_C_gpu_result, d_A, d_B, d_C, N, "CUDA TRIVIAL", &t_triv_avg, &t_triv_unc, num_runs);
    
    int triv_ok = verify(h_C_serial, h_C_gpu_result, N);

    cudaMemset(d_C, 0, N * N * sizeof(double));
    benchmark(launch_cuda_optimized, h_A, h_B, h_C_gpu_result, d_A, d_B, d_C, N, "CUDA OPTIMIZED", &t_opt_avg, &t_opt_unc, num_runs);
    
    int opt_ok = verify(h_C_serial, h_C_gpu_result, N);

    printf("\n--- RESULTS SUMMARY ---\n");
    printf("Serial time:    %.4fs\n", t_serial_avg);
    printf("Trivial avg:    %.4fs (Unc: %.4fs) -> Speedup: %.2fx\n", t_triv_avg, t_triv_unc, t_serial_avg / t_triv_avg);
    printf("Optimized avg:  %.4fs (Unc: %.4fs) -> Speedup: %.2fx\n", t_opt_avg, t_opt_unc, t_serial_avg / t_opt_avg);
    printf("Opt vs Triv speedup: %.2fx\n", t_triv_avg / t_opt_avg);

    printf("\n--- VERIFICATION ---\n");
    printf("Trivial Correctness:   %s\n", triv_ok ? "OK" : "FAIL");
    printf("Optimized Correctness: %s\n", opt_ok ? "OK" : "FAIL");

    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C_gpu_result); free(h_C_serial);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}