#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

float launch_cuda_integration(float start, float end, int num_steps);

#define TOTAL_START 0
#define TOTAL_END 1000
#define TOTAL_STEPS 10000000 

int main(int argc, char **argv) {
    int num_threads = omp_get_max_threads();
    if (argc > 1) num_threads = atoi(argv[1]);

    omp_set_num_threads(num_threads);

    double start_time = omp_get_wtime();
    float global_integral = 0;

    printf("Starting integration with %d threads (OpenMP + CUDA)...\n", num_threads);

    #pragma omp parallel reduction(+:global_integral)
    {
        int tid = omp_get_thread_num();

        float range_len = (TOTAL_END - TOTAL_START);
        float chunk_size = range_len / num_threads;
        
        float local_start = TOTAL_START + tid * chunk_size;
        float local_end = local_start + chunk_size;
        int local_steps = TOTAL_STEPS / num_threads;

        printf("Thread %d: processing range [%.2f, %.2f]\n", tid, local_start, local_end);

        float local_area = launch_cuda_integration(local_start, local_end, local_steps);
        
        global_integral += local_area;
    }

    double end_time = omp_get_wtime();

    printf("\nTotal Integral: %f\n", global_integral);
    printf("Time: %f s\n", end_time - start_time);

    return 0;
}