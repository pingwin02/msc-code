#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "numgen.c"


int main(int argc,char **argv) {


  Args ins__args;
  parseArgs(&ins__args, &argc, argv);

  //set number of threads
  omp_set_num_threads(ins__args.n_thr);
  
  //program input argument
  long inputArgument = ins__args.arg; 
  unsigned long int *numbers = (unsigned long int*)malloc(inputArgument * sizeof(unsigned long int));
  numgen(inputArgument, numbers);

  struct timeval ins__tstart, ins__tstop;
  gettimeofday(&ins__tstart, NULL);
  
  // run your computations here (including OpenMP stuff)

  int count_primes = 0;
  int chunk_size = 100;

  #pragma omp parallel for schedule(guided, chunk_size) reduction(+:count_primes)
  for (long i = 0; i < inputArgument; i++) {
    unsigned long int num = numbers[i];
    if (num < 2) continue;
    int is_prime = 1;

    for (unsigned long int j = 2; j * j <= num; j++) {
      if (num % j == 0) {
        is_prime = 0;
        break;
      }
    }

    count_primes += is_prime;
  }

  printf("\nTotal prime numbers found: %d\n", count_primes);
  free(numbers);
  
  // synchronize/finalize your computations
  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);

}
