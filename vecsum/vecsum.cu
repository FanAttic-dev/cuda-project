#include <stdio.h>
#include <stdlib.h>

#define BLOCKED

#define N 8192
#define BLOCK_SIZE 128

__global__ void vecsum(float *V, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	V[i] = V[i] + V[i + blockDim.x];
}

__global__ void vecsum_blocked(float *V)
{
	extern __shared__ float Vs[];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	Vs[threadIdx.x]	= V[i];
	__syncthreads();

	for (int s = 1; s < blockDim.x; s *= 2) {
		if (threadIdx.x % (s * 2) == 0)
			Vs[threadIdx.x] += Vs[threadIdx.x + s];
		__syncthreads();
	}

	if (threadIdx.x == 0)
		V[blockIdx.x * blockDim.x] = Vs[0];
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
	float sum = 0;

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

#ifdef NON_BLOCKED
	for (int n = N/2; n > 0; n /= 2) {
		vecsum<<<1, n>>>(dV, n);
	}
#elif defined(BLOCKED)
	vecsum_blocked<<<N/BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(dV);
#endif

	cudaEventRecord(stop);

	cudaMemcpy(hV, dV, vecSizeBytes, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

#ifdef NON_BLOCKED
	sum = hV[0];
#elif defined(BLOCKED)
	for (int i = 0; i < N; i += BLOCK_SIZE)
		sum += hV[i];
#endif
	printf("N = %d\n", N);
	printf("Sum: %f\n", sum);
	printf("Elapsed time: %f ms\n", milliseconds);
	printf("GPU performance: %f megaevals/s\n", float(N*N)/milliseconds/1000.f);

	status = EXIT_SUCCESS;

cleanup:
	cudaFree(dV);
	free(hV);

	return status;
}
