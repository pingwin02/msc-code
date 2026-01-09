#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>

#define DEFAULT_N 10000000
#define MAX_RAND 100

int main(int argc, char **argv) {
    int threadnum = omp_get_max_threads();
    int N = DEFAULT_N;

    if (argc > 1) {
        int t = atoi(argv[1]);
        if (t > 0) threadnum = t;
    }
    if (argc > 2) {
        int n_arg = atoi(argv[2]);
        if (n_arg > 0) N = n_arg;
    }

    omp_set_num_threads(threadnum);

    int *array = (int*)malloc(N * sizeof(int));
    int *histogram = (int*)calloc(MAX_RAND, sizeof(int));

    if (!array || !histogram) return 1;

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        array[i] = rand() % MAX_RAND;
    }

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        int *local_hist = (int*)calloc(MAX_RAND, sizeof(int));

        #pragma omp for
        for (int i = 0; i < N; i++) {
            local_hist[array[i]]++;
        }

        for (int i = 0; i < MAX_RAND; i++) {
            #pragma omp atomic
            histogram[i] += local_hist[i];
        }

        free(local_hist);
    }

    double end = omp_get_wtime();

    printf("Threads: %d, Elements: %d\n", threadnum, N);
    printf("Time: %f s\n", end - start);

    long long sum_check = 0;
    for (int i = 0; i < MAX_RAND; i++) {
        sum_check += histogram[i];
    }

    printf("Sum: %lld (Expected: %d)\n", sum_check, N);
    printf("Status: %s\n", (sum_check == N) ? "OK" : "FAIL");

    free(array);
    free(histogram);

    return 0;
}
