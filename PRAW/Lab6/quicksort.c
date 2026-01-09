#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define DEFAULT_N 1000000
#define MAX_DEPTH 4

void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quicksort_serial(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort_serial(arr, low, pi - 1);
        quicksort_serial(arr, pi + 1, high);
    }
}

void quicksort_parallel(int arr[], int low, int high, int depth) {
    if (low < high) {
        int pi = partition(arr, low, high);

        if (depth < MAX_DEPTH) {
            #pragma omp parallel sections
            {
                #pragma omp section
                quicksort_parallel(arr, low, pi - 1, depth + 1);

                #pragma omp section
                quicksort_parallel(arr, pi + 1, high, depth + 1);
            }
        } else {
            quicksort_serial(arr, low, pi - 1);
            quicksort_serial(arr, pi + 1, high);
        }
    }
}

int main(int argc, char **argv) {
    int threads = omp_get_max_threads();
    int N = DEFAULT_N;

    if (argc > 1) {
        int t = atoi(argv[1]);
        if (t > 0) threads = t;
    }

    if (argc > 2) {
        int n_arg = atoi(argv[2]);
        if (n_arg > 0) N = n_arg;
    }

    omp_set_num_threads(threads);
    omp_set_nested(1);
    omp_set_dynamic(0);

    int *arr = (int*)malloc(N * sizeof(int));
    if (!arr) return 1;

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 10000;
    }

    printf("Threads: %d, Elements: %d\n", threads, N);
    
    double start = omp_get_wtime();
    quicksort_parallel(arr, 0, N - 1, 0);
    double end = omp_get_wtime();

    printf("Time: %f s\n", end - start);

    int sorted = 1;
    for (int i = 0; i < N - 1; i++) {
        if (arr[i] > arr[i+1]) {
            sorted = 0;
            break;
        }
    }

    if (sorted) {
        printf("Status: Sorted\n");
    } else {
        printf("Status: FAIL\n");
    }

    free(arr);
    return 0;
}