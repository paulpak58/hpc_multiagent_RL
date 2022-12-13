#include <iostream>
#include <cuda.h>
#include <random>
#include "random.cuh"
#include "alpha_beta.cuh"
using namespace std;

int main(int argc,char *argv[]){
	int n = atoi(argv[1]);
	int threads_per_block = atoi(argv[2]);

	// Timing Initialization
	cudaEvent_t startEvent,stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	float elapsedTime;

	// Random Initialization
	random_device entropy_source;
	mt19937_64 generator(entropy_source());
	uniform_real_distribution<float> dist(-1.0,1.0);

	// Create arrays in managed memory
	float *A,*B;
	cudaMallocManaged(&A,n*sizeof(float));
	cudaMallocManaged(&B,n*sizeof(float));
	for(int i=0;i<n;++i)
		A[i] = dist(generator);

	// Start time and call scan to conduct an inclusive scan
	cudaEventRecord(startEvent,0);

	// ALPHA BETA ALGO

	cudaEventRecord(stopEvent,0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime,startEvent,stopEvent);

 	printf("%f\n",B[n-1]);
	printf("%f\n",elapsedTime);

	// Clean up and free memory
	cudaFree(A);
	cudaFree(B);
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);
	return 0;
}
