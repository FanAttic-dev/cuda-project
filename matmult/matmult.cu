#include <stdio.h>
#include <stdlib.h>

#define N 8

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

int main(void)
{
	size_t blockSize = N;
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

	// fill the matrix
	int i, j;
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			hA[i * N + j] = 10.f * (float) rand() / (float) RAND_MAX;
			hB[i * N + j] = 10.f * (float) rand() / (float) RAND_MAX;
		}
	}

	cudaMemcpy(dA, hA, matrixSizeBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, matrixSizeBytes, cudaMemcpyHostToDevice);

	matMult<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, N);
	cudaDeviceSynchronize();

	cudaMemcpy(hC, dC, matrixSizeBytes, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	printMatrix("hA", hA, N);
	puts("+");
	printMatrix("hB", hB, N);
	puts("=");
	printMatrix("hC", hC, N);

	status = EXIT_SUCCESS;

cleanup:
	cudaFree(dA); cudaFree(dB); cudaFree(dC);
	free(hA); free(hB); free(hC);

	return status;
}
