#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
#include "numgen.c"

int count_local_primes(unsigned long int *numbers, long size) {
  int local_prime_count = 0;
#pragma omp parallel for schedule(guided, 100) reduction(+ : local_prime_count)
  for (long i = 0; i < size; i++) {
    unsigned long int num = numbers[i];
    int is_prime = 1;
    if (num < 2)
      is_prime = 0;
    else if (num == 2)
      is_prime = 1;
    else if (num % 2 == 0)
      is_prime = 0;
    else {
      for (unsigned long int j = 3; j * j <= num; j += 2) {
        if (num % j == 0) {
          is_prime = 0;
          break;
        }
      }
    }
    if (is_prime) local_prime_count++;
  }
  return local_prime_count;
}

int main(int argc, char **argv) {
  Args ins__args;
  parseArgs(&ins__args, &argc, argv);

  // set number of threads
  omp_set_num_threads(ins__args.n_thr);

  // program input argument
  long inputArgument = ins__args.arg;

  struct timeval ins__tstart, ins__tstop;

  int threadsupport;
  int myrank, nproc;
  unsigned long int *numbers;

  // Initialize MPI with desired support for multithreading -- state your desired support level
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &threadsupport);

  if (threadsupport < MPI_THREAD_FUNNELED) {
    printf("\nThe implementation does not support MPI_THREAD_FUNNELED, it supports level %d\n", threadsupport);
    MPI_Finalize();
    return -1;
  }

  // obtain my rank
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  // and the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (!myrank) {
    gettimeofday(&ins__tstart, NULL);
    numbers = (unsigned long int *)malloc(inputArgument * sizeof(unsigned long int));
    numgen(inputArgument, numbers);
  }

  // run your computations here (including MPI communication and OpenMP stuff)

  long base_chunk = inputArgument / nproc;
  long rem = inputArgument % nproc;
  long local_size = base_chunk + (myrank == 0 ? rem : 0);

  if (myrank == 0) {
    for (int r = 1; r < nproc; r++) {
      MPI_Send(numbers + rem + r * base_chunk, base_chunk, MPI_UNSIGNED_LONG, r, 0, MPI_COMM_WORLD);
    }
  } else {
    numbers = malloc(local_size * sizeof(unsigned long int));
    MPI_Recv(numbers, local_size, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  int local_prime_count = count_local_primes(numbers, local_size);

  if (myrank == 0) {
    int global_prime_count = local_prime_count;
    int temp;
    for (int r = 1; r < nproc; r++) {
      MPI_Recv(&temp, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      global_prime_count += temp;
    }
    printf("Total prime numbers found: %d\n", global_prime_count);
  } else {
    MPI_Send(&local_prime_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  // synchronize/finalize your computations

  if (!myrank) {
    gettimeofday(&ins__tstop, NULL);
    ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
  }

  MPI_Finalize();
}