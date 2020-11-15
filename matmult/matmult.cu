#include <stdio.h>
#include <stdlib.h>

#define N 2048
#define THREADS_PER_BLOCK 1024

__global__ void matMult(const float *A, const float *B, float *C, int n)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row >= n || col >= n)
		return;

	float tmp = 0.f;
	int i;
	for (i = 0; i < n; ++i)
		tmp += A[row * n + i] * B[i * n + col];

	C[row * n + col] = tmp;
}

void printMatrix(const char *name, float *M, size_t n)
{
	printf("%s:\n", name);

	int i, j;
	for (i = 0; i < n; ++i) {
		for (j = 0; j < n; ++j) {
			printf("%8.2f ", M[i * n + j]);
		}
		printf("\n");
	}
}

void fillMatrices(float *A, float *B)
{	
	int i, j;
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			A[i * N + j] = 10.f * (float) rand() / (float) RAND_MAX;
			B[i * N + j] = 10.f * (float) rand() / (float) RAND_MAX;
		}
	}
}

int main(void)
{
	size_t blockSize = sqrt(THREADS_PER_BLOCK);
	dim3 threadsPerBlock(blockSize, blockSize);

	int nBlocks = ceil(N/blockSize);
	dim3 blocksPerGrid(nBlocks, nBlocks);

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

	size_t matrixSizeBytes = N * N * sizeof(float);
	int status = EXIT_FAILURE;
	float *hA, *hB, *hC;
	float *dA, *dB, *dC;
	hA = (float *) malloc(matrixSizeBytes);
	hB = (float *) malloc(matrixSizeBytes);
	hC = (float *) malloc(matrixSizeBytes);

	cudaMalloc(&dA, matrixSizeBytes);
	cudaMalloc(&dB, matrixSizeBytes);
	cudaMalloc(&dC, matrixSizeBytes);

	if (!hA || !hB || !hC || !dA || !dB || !dC) {
		fprintf(stderr, "Could not allocate memory!\n");
		goto cleanup;
	}
	
	fillMatrices(hA, hB);

	cudaMemcpy(dA, hA, matrixSizeBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, matrixSizeBytes, cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	matMult<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, N);
	cudaEventRecord(stop);
//	cudaDeviceSynchronize();

	cudaMemcpy(hC, dC, matrixSizeBytes, cudaMemcpyDeviceToHost);
//	cudaDeviceSynchronize();

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

#if 0
	printMatrix("hA", hA, N);
	puts("+");
	printMatrix("hB", hB, N);
	puts("=");
	printMatrix("hC", hC, N);
#endif

	printf("Calculation status: %s\n", hC[0] != 0 ? "success" : "failed");
	printf("Block size: %lu x %lu (%d threads per block)\n", blockSize, blockSize, THREADS_PER_BLOCK);
	printf("Elapsed time: %f ms\n", milliseconds);
	printf("GPU performance: %f megaevals/s\n", float(N*N)/milliseconds/1000.f);

	status = EXIT_SUCCESS;

cleanup:
	cudaFree(dA); cudaFree(dB); cudaFree(dC);
	free(hA); free(hB); free(hC);

	return status;
}
