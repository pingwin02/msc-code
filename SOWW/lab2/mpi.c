#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "numgen.c"
#include "utility.h"

int is_prime(unsigned long int num) {
  if (num < 2) return 0;
  if (num == 2) return 1;
  if (num % 2 == 0) return 0;
  for (unsigned long int i = 3; i * i <= num; i += 2) {
    if (num % i == 0) return 0;
  }
  return 1;
}

int main(int argc, char** argv) {
  Args ins__args;
  parseArgs(&ins__args, &argc, argv);

  // program input argument
  long inputArgument = ins__args.arg;

  struct timeval ins__tstart, ins__tstop;
  int myrank, nproc;
  unsigned long int* numbers;

  MPI_Init(&argc, &argv);

  // obtain my rank
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  // and the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (!myrank) {
    printf("nproc: %d\n", nproc);
    gettimeofday(&ins__tstart, NULL);
    numbers = (unsigned long int*)malloc(inputArgument * sizeof(unsigned long int));
    numgen(inputArgument, numbers);
  }

  // run your computations here (including MPI communication)

  long base_chunk_size = inputArgument / nproc;
  long remainder = inputArgument % nproc;

  long local_size = base_chunk_size + (myrank == 0 ? remainder : 0);

  if (myrank) {
    numbers = (unsigned long int*)malloc(local_size * sizeof(unsigned long int));
    MPI_Recv(numbers, local_size, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    for (int rank = 1; rank < nproc; rank++) {
      MPI_Send(numbers + rank * base_chunk_size + remainder, base_chunk_size, MPI_UNSIGNED_LONG, rank, 0, MPI_COMM_WORLD);
    }
  }

  int local_prime_count = 0;

  for (int i = 0; i < local_size; i++) {
    if (is_prime(numbers[i])) {
      local_prime_count++;
    }
  }

  if (myrank) {
    MPI_Send(&local_prime_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  } else {
    int global_prime_count = local_prime_count;
    int temp_count;

    for (int rank = 1; rank < nproc; rank++) {
      MPI_Recv(&temp_count, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      global_prime_count += temp_count;
    }

    printf("Total prime numbers found: %d\n", global_prime_count);
  }

  // synchronize/finalize your computations
  
  if (!myrank) {
    gettimeofday(&ins__tstop, NULL);
    ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
  }

  free(numbers);
  MPI_Finalize();
}