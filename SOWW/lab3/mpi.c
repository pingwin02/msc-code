#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "numgen.c"
#include "utility.h"

#define RANGESIZE 100000
#define DATA 0
#define RESULT 1

int is_prime(unsigned long int num) {
	if (num < 2) return 0;
	if (num == 2) return 1;
	if (num % 2 == 0) return 0;
	for (unsigned long int i = 3; i * i <= num; i += 2) {
		if (num % i == 0) return 0;
	}
	return 1;
}

int CountPrimes(unsigned long int *arr, long start, long end) {
	int count = 0;
	for (long i = start; i < end; i++) {
		if (is_prime(arr[i])) count++;
	}
	return count;
}

int main(int argc, char** argv) {
	Args ins__args;
	parseArgs(&ins__args, &argc, argv);

	long a = 0, b = ins__args.arg;

	struct timeval ins__tstart, ins__tstop;
	int myrank, proccount;
	unsigned long int* numbers = NULL;

	// Initialize MPI

	MPI_Init(&argc, &argv);

	// find out my rank
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	// find out the number of processes in MPI_COMM_WORLD
	MPI_Comm_size(MPI_COMM_WORLD, &proccount);

	if (myrank == 0) {
		printf("\n\nproccount: %d\n", proccount);
		fflush(stdout);
		gettimeofday(&ins__tstart, NULL);
		numbers = (unsigned long int*)malloc(b * sizeof(unsigned long int));
		numgen(b, numbers);

		if (!numbers) {
			printf("\nNot enough memory");
			MPI_Finalize();
			return -1;
		}
	}

	MPI_Request *requests;
	int requestcompleted;
	long *ranges;
	long range[2];
	long result = 0;
	long *resulttemp;
	int sentcount = 0;
	int recvcount = 0;
	int i;
	MPI_Status status;

	if (proccount < 2) {
		printf("Run with at least 2 processes");
		MPI_Finalize();
		return -1;
	}

	if (((b - a) / RANGESIZE) < 2 * (proccount - 1)) {
		printf("More subranges needed");
		MPI_Finalize();
		return -1;
	}

	// now the master will distribute the data and slave processes will perform computations
	if (myrank == 0) {
		for (i = 1; i < proccount; i++) {
			MPI_Send(&b, 1, MPI_LONG, i, DATA, MPI_COMM_WORLD);
			MPI_Send(numbers, b, MPI_UNSIGNED_LONG, i, DATA, MPI_COMM_WORLD);
		}
		requests = (MPI_Request * ) malloc(3 * (proccount - 1) * sizeof(MPI_Request));

		if (!requests) {
			printf("\nNot enough memory");
			MPI_Finalize();
			return -1;
		}

		ranges = (long * ) malloc(4 * (proccount - 1) * sizeof(long));
		if (!ranges) {
			printf("\nNot enough memory");
			MPI_Finalize();
			return -1;
		}

		resulttemp = (long * ) malloc((proccount - 1) * sizeof(long));
		if (!resulttemp) {
			printf("\nNot enough memory");
			MPI_Finalize();
			return -1;
		}

		range[0] = a;

		// first distribute some ranges to all slaves
		for (i = 1; i < proccount; i++) {
			range[1] = range[0] + RANGESIZE;

			// send it to process i
			MPI_Send(range, 2, MPI_LONG, i, DATA, MPI_COMM_WORLD);
			sentcount++;
			range[0] = range[1];
		}

		// the first proccount requests will be for receiving, the latter ones for sending
		for (i = 0; i < 2 * (proccount - 1); i++)
			requests[i] = MPI_REQUEST_NULL;

		// start receiving for results from the slaves
		for (i = 1; i < proccount; i++)
			MPI_Irecv(&(resulttemp[i - 1]), 1, MPI_LONG, i, RESULT, MPI_COMM_WORLD, &(requests[i - 1]));

		// start sending new data parts to the slaves
		for (i = 1; i < proccount; i++) {
			range[1] = range[0] + RANGESIZE;
			if (range[1] > b) range[1] = b;
			ranges[2 * i - 2] = range[0];
			ranges[2 * i - 1] = range[1];

			// send it to process i
			MPI_Isend(&(ranges[2 * i - 2]), 2, MPI_LONG, i, DATA, MPI_COMM_WORLD,
			          &(requests[proccount - 2 + i]));
			sentcount++;
			range[0] = range[1];
		}

		while (range[1] < b) {
			// wait for completion of any of the requests
			MPI_Waitany(2 * proccount - 2, requests, &requestcompleted, MPI_STATUS_IGNORE);

			// if it is a result then send new data to the process
			// and add the result
			if (requestcompleted < (proccount - 1)) {
				result += resulttemp[requestcompleted];
				recvcount++;

				// first check if the send has terminated
				MPI_Wait(&(requests[proccount - 1 + requestcompleted]), MPI_STATUS_IGNORE);

				// now send some new data portion to this process
				range[1] = range[0] + RANGESIZE;
				if (range[1] > b)
					range[1] = b;

				ranges[2 * requestcompleted] = range[0];
				ranges[2 * requestcompleted + 1] = range[1];

				MPI_Isend(&(ranges[2 * requestcompleted]), 2, MPI_LONG,
				          requestcompleted + 1, DATA, MPI_COMM_WORLD,
				          &(requests[proccount - 1 + requestcompleted]));
				sentcount++;
				range[0] = range[1];

				// now issue a corresponding recv
				MPI_Irecv(&(resulttemp[requestcompleted]), 1, MPI_LONG,
				          requestcompleted + 1, RESULT, MPI_COMM_WORLD,
				          &(requests[requestcompleted]));
			}
		}
		// now send the FINISHING ranges to the slaves
		// shut down the slaves
		range[0] = range[1];
		for (i = 1; i < proccount; i++) {
			ranges[2 * i - 4 + 2 * proccount] = range[0];
			ranges[2 * i - 3 + 2 * proccount] = range[1];
			MPI_Isend(range, 2, MPI_LONG, i, DATA, MPI_COMM_WORLD, &
			          (requests[2 * proccount - 3 + i]));
		}

		// now receive results from the processes - that is finalize the pending requests
		MPI_Waitall(3 * proccount - 3, requests, MPI_STATUSES_IGNORE);

		// now simply add the results
		for (i = 0; i < (proccount - 1); i++)
			result += resulttemp[i];
		// now receive results for the initial sends
		for (i = 0; i < (proccount - 1); i++) {

			MPI_Recv(&(resulttemp[i]), 1, MPI_LONG, i + 1, RESULT, MPI_COMM_WORLD, &status);
			result += resulttemp[i];
			recvcount++;

		}
		// now display the result
		printf("\nTotal prime numbers found: %ld\n", result);
		fflush(stdout);
	} else //slave
	{
		MPI_Recv(&b, 1, MPI_LONG, 0, DATA, MPI_COMM_WORLD, &status);
		numbers = malloc(sizeof(unsigned long int) * b);
		if (!numbers) {
			printf("\nNot enough memory");
			MPI_Finalize();
			return -1;
		}
		MPI_Recv(numbers, b, MPI_UNSIGNED_LONG, 0, DATA, MPI_COMM_WORLD, &status);

		requests = (MPI_Request *) malloc(2 * sizeof(MPI_Request));

		if (!requests) {
			printf("\nNot enough memory");
			MPI_Finalize();
			return -1;
		}

		requests[0] = requests[1] = MPI_REQUEST_NULL;
		ranges = (long *) malloc(2 * sizeof(long));

		if (!ranges) {
			printf("\nNot enough memory");
			MPI_Finalize();
			return -1;
		}

		resulttemp = (long *) malloc(2 * sizeof(long));

		if (!resulttemp) {
			printf("\nNot enough memory");
			MPI_Finalize();
			return -1;
		}

		// first receive the initial data
		MPI_Recv(range, 2, MPI_LONG, 0, DATA, MPI_COMM_WORLD, &status);

		while (range[0] < range[1]) {
			// if there is some data to process
			// before computing the next part start receiving a new data part
			MPI_Irecv(ranges, 2, MPI_LONG, 0, DATA, MPI_COMM_WORLD, &(requests[0]));

			// compute my part
			resulttemp[1] = CountPrimes(numbers, range[0], range[1]);

			// now finish receiving the new part
			// and finish sending the previous results back to the master
			MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
			range[0] = ranges[0];
			range[1] = ranges[1];
			resulttemp[0] = resulttemp[1];

			// and start sending the results back
			MPI_Isend(&resulttemp[0], 1, MPI_LONG, 0, RESULT, MPI_COMM_WORLD, &(requests[1]));
		}
		// now finish sending the last results to the master
		MPI_Wait(&(requests[1]), MPI_STATUS_IGNORE);
	}

	if (!myrank) {
		gettimeofday(&ins__tstop, NULL);
		ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
	}

	// Shut down MPI
	MPI_Finalize();

	return 0;
}
