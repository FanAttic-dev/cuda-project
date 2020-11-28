#include <stdio.h>
#include <stdlib.h>

#define N 2048
#define BLOCK_SIZE 32

__global__ void vecsum(float *V, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	V[i] = V[i] + V[i + blockDim.x];
}

void fillVector(float *v)
{	
	float sum = 0.f;
	for (int i = 0; i < N; ++i) {
		sum += i;
	}
	for (int i = 0; i < N; ++i) {
		v[i] = (float) i / sum;
	}
}

int main(void)
{
	int device = 0;
	if (cudaSetDevice(device) != cudaSuccess) {
		fprintf(stderr, "Could not set CUDA device!\n");
		return EXIT_FAILURE;
	}

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);
	printf("Using device %d: \"%s\"\n", device, deviceProp.name);	

	// setup timers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	int status = EXIT_FAILURE;
	float *hV;
	float *dV;
	size_t vecSizeBytes = N * sizeof(float);

	hV = (float *) malloc(vecSizeBytes);
	cudaMalloc(&dV, vecSizeBytes);

	if (!hV || !dV) {
		fprintf(stderr, "Could not allocate memory!\n");
		goto cleanup;
	}
	
	fillVector(hV);

	cudaMemcpy(dV, hV, vecSizeBytes, cudaMemcpyHostToDevice);

	cudaEventRecord(start);

	for (int n = N/2; n > 0; n /= 2) {
		vecsum<<<1, n>>>(dV, n);
	}

	cudaEventRecord(stop);

	cudaMemcpy(hV, dV, vecSizeBytes, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("N = %d\n", N);
	printf("Sum: %f\n", hV[0]);
	printf("Elapsed time: %f ms\n", milliseconds);
	printf("GPU performance: %f megaevals/s\n", float(N*N)/milliseconds/1000.f);

	status = EXIT_SUCCESS;

cleanup:
	cudaFree(dV);
	free(hV);

	return status;
}
