#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

typedef void (*matmul_func)(double*, double*, double*, int);

void init_matrix(double** M, int N) {
    *M = (double*)malloc(N * N * sizeof(double));
    if (*M == NULL) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(1);
    }
    for (int i = 0; i < N * N; i++) {
        (*M)[i] = (double)rand() / RAND_MAX;
    }
}

void matmul_serial(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void matmul_omp_trivial(double* A, double* B, double* C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void matmul_omp_optimized(double* A, double* B, double* C, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N * N; i++) C[i] = 0;

    #pragma omp parallel for schedule(static)
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
    double sum = 0;
    double min_time = 1e9;
    double max_time = 0;

    printf("Benchmarking %s (%d runs)...\n", name, num_runs); fflush(stdout);

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
}

int verify(double* C1, double* C2, int N) {
    double epsilon = 1e-7;
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
    int threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    int num_runs = 10;

    omp_set_num_threads(threads);
    printf("=== Matrix Multiplication %dx%d | Threads: %d | Runs per test: %d ===\n", N, N, threads, num_runs);

    double *A, *B, *C_serial, *C_triv, *C_opt;
    
    printf("Initializing matrices...\n");
    init_matrix(&A, N);
    init_matrix(&B, N);
    
    C_serial = (double*)calloc(N * N, sizeof(double));
    C_triv = (double*)calloc(N * N, sizeof(double));
    C_opt = (double*)calloc(N * N, sizeof(double));

    if (C_serial == NULL || C_triv == NULL || C_opt == NULL) {
        fprintf(stderr, "Failed to allocate C matrices\n");
        return 1;
    }

    double t_serial_avg;
    double t_triv_avg, t_triv_unc;
    double t_opt_avg, t_opt_unc;

    printf("Serial version (baseline)...\n"); fflush(stdout);
    double t0 = omp_get_wtime();
    matmul_serial(A, B, C_serial, N);
    t_serial_avg = omp_get_wtime() - t0;

    benchmark(matmul_omp_trivial, A, B, C_triv, N, "OpenMP TRIVIAL", &t_triv_avg, &t_triv_unc, num_runs);

    benchmark(matmul_omp_optimized, A, B, C_opt, N, "OpenMP OPTIMIZED", &t_opt_avg, &t_opt_unc, num_runs);

    printf("\n--- RESULTS SUMMARY ---\n");
    printf("Serial time:    %.4fs\n", t_serial_avg);
    printf("Trivial avg:    %.4fs (Unc: %.4fs) -> Speedup: %.2fx\n", t_triv_avg, t_triv_unc, t_serial_avg / t_triv_avg);
    printf("Optimized avg:  %.4fs (Unc: %.4fs) -> Speedup: %.2fx\n", t_opt_avg, t_opt_unc, t_serial_avg / t_opt_avg);
    printf("Opt vs Triv speedup: %.2fx\n", t_triv_avg / t_opt_avg);

    printf("\n--- VERIFICATION ---\n");
    printf("Trivial Correctness:   %s\n", verify(C_serial, C_triv, N) ? "OK" : "FAIL");
    printf("Optimized Correctness: %s\n", verify(C_serial, C_opt, N) ? "OK" : "FAIL");

    free(A); free(B); free(C_serial); free(C_triv); free(C_opt);
    return 0;
}