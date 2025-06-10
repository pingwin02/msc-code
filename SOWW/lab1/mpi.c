#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

int main(int argc,char **argv) {

  Args ins__args;
  parseArgs(&ins__args, &argc, argv);

  // program input argument
  long inputArgument = ins__args.arg;

  struct timeval ins__tstart, ins__tstop;

  int myrank, nproc;
  
  MPI_Init(&argc, &argv);

  // obtain my rank
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  // and the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (!myrank)
      gettimeofday(&ins__tstart, NULL);

  // run your computations here (including MPI communication)

  int is_prime = 1;

  if (inputArgument < 2) {
      is_prime = 0;
  } else {
      long start = 3 + 2 * myrank;
      long step = 2 * nproc;
      for (long i = start; i * i <= inputArgument; i += step) {
          if (inputArgument % i == 0) {
              is_prime = 0;
              break;
          }
      }
  }

  int global_is_prime;
  MPI_Reduce(&is_prime, &global_is_prime, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);

  if (myrank == 0) {
      if (global_is_prime)
          printf("%lu is a prime number.\n", inputArgument);
      else
          printf("%lu is not a prime number.\n", inputArgument);
  }

  // synchronize/finalize your computations

  if (!myrank) {
    gettimeofday(&ins__tstop, NULL);
    ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
  }
  
  MPI_Finalize();

}
