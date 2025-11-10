#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

typedef void (*matmul_func)(double*, double*, double*, int);

void init_matrix(double** M, int N) {
    *M = (double*)malloc(N * N * sizeof(double));
    for (int i = 0; i < N * N; i++) {
        (*M)[i] = (double)rand() / RAND_MAX;
    }
}

void matmul_serial(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_omp_trivial(double* A, double* B, double* C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_omp_optimized(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N * N; i++) C[i] = 0.0;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a_val = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
}

void benchmark(matmul_func func, double* A, double* B, double* C, int N, const char* name, double* avg_time, double* uncertainty, int num_runs) {
    double sum = 0.0;
    double min_time = 1e9;
    double max_time = 0.0;

    printf("Benchmarking %s (%d runs)... ", name, num_runs); fflush(stdout);

    for (int i = 0; i < num_runs; i++) {
        double start = omp_get_wtime();
        func(A, B, C, N);
        double end = omp_get_wtime();
        double time = end - start;
        
        sum += time;
        if (time < min_time) min_time = time;
        if (time > max_time) max_time = time;
    }

    *avg_time = sum / num_runs;
    *uncertainty = (max_time - min_time) / 2.0;

    printf("Avg: %.4fs (Unc: %.4fs)\n", *avg_time, *uncertainty);
}

int verify(double* C1, double* C2, int N) {
    double epsilon = 1e-7;
    for (int i = 0; i < N * N; i++) {
        if (fabs(C1[i] - C2[i]) > epsilon) return 0;
    }
    return 1;
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    int num_runs = 10;

    omp_set_num_threads(threads);
    printf("=== Matrix Multiplication %dx%d | Threads: %d | Runs per test: %d ===\n", N, N, threads, num_runs);

    double *A, *B, *C_serial, *C_triv, *C_opt;
    init_matrix(&A, N);
    init_matrix(&B, N);
    C_serial = (double*)calloc(N * N, sizeof(double));
    C_triv = (double*)malloc(N * N * sizeof(double));
    C_opt = (double*)malloc(N * N * sizeof(double));

    double t_serial_avg;
    double t_triv_avg, t_triv_unc;
    double t_opt_avg, t_opt_unc;

    printf("1. Serial version (baseline)... "); fflush(stdout);
    double t0 = omp_get_wtime();
    matmul_serial(A, B, C_serial, N);
    t_serial_avg = omp_get_wtime() - t0;
    printf("Done (single run): %.4fs\n", t_serial_avg);

    benchmark(matmul_omp_trivial, A, B, C_triv, N, "OpenMP TRIVIAL", &t_triv_avg, &t_triv_unc, num_runs);

    benchmark(matmul_omp_optimized, A, B, C_opt, N, "OpenMP OPTIMIZED", &t_opt_avg, &t_opt_unc, num_runs);

    printf("\n--- RESULTS SUMMARY ---\n");
    printf("Serial time:    %.4fs\n", t_serial_avg);
    printf("Trivial avg:    %.4fs (Unc: %.4fs) -> Speedup: %.2fx\n", t_triv_avg, t_triv_unc, t_serial_avg / t_triv_avg);
    printf("Optimized avg:  %.4fs (Unc: %.4fs) -> Speedup: %.2fx\n", t_opt_avg, t_opt_unc, t_serial_avg / t_opt_avg);
    printf("Opt vs Triv speedup: %.2fx\n", t_triv_avg / t_opt_avg);

    printf("\n--- VERIFICATION ---\n");
    printf("Trivial Correctness: %s\n", verify(C_serial, C_triv, N) ? "OK" : "FAIL");
    printf("Optimized Correctness: %s\n", verify(C_serial, C_opt, N) ? "OK" : "FAIL");

    free(A); free(B); free(C_serial); free(C_triv); free(C_opt);
    return 0;
}