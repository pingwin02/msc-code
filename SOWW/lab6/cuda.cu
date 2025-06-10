#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "numgen.c"

__device__ bool isPrime(unsigned long int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (unsigned long int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

__global__ void checkPrimes(const unsigned long int *numbers, int *primeFlags, long size) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primeFlags[idx] = isPrime(numbers[idx]) ? 1 : 0;
    }
}

int main(int argc, char **argv)
{

  Args ins__args;
  parseArgs(&ins__args, &argc, argv);

  // program input argument
  long inputArgument = ins__args.arg;
  unsigned long int *numbers = (unsigned long int *)malloc(inputArgument * sizeof(unsigned long int));
  numgen(inputArgument, numbers);

  struct timeval ins__tstart, ins__tstop;
  gettimeofday(&ins__tstart, NULL);

  // run your CUDA kernel(s) here

  unsigned long int *d_numbers;
  int *d_primeFlags;
  int *h_primeFlags = (int *)malloc(inputArgument * sizeof(int));

  cudaMalloc((void **)&d_numbers, inputArgument * sizeof(unsigned long int));
  cudaMalloc((void **)&d_primeFlags, inputArgument * sizeof(int));

  cudaMemcpy(d_numbers, numbers, inputArgument * sizeof(unsigned long int), cudaMemcpyHostToDevice);

  int threadsPerBlock = 1024;
  int blocksPerGrid = (inputArgument + threadsPerBlock - 1) / threadsPerBlock;

  checkPrimes<<<blocksPerGrid, threadsPerBlock>>>(d_numbers, d_primeFlags, inputArgument);
  cudaDeviceSynchronize();

  cudaMemcpy(h_primeFlags, d_primeFlags, inputArgument * sizeof(int), cudaMemcpyDeviceToHost);

  long primeCount = 0;
  for (long i = 0; i < inputArgument; i++)
  {
    primeCount += h_primeFlags[i];
  }

  printf("Number of primes: %ld\n", primeCount);

  // synchronize/finalize your CUDA computations

  cudaFree(d_numbers);
  cudaFree(d_primeFlags);
  free(h_primeFlags);
  free(numbers);

  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
}
