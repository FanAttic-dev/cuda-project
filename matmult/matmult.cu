#include <stdio.h>
#include <stdlib.h>

#define N 2048
#define TILE_SIZE 32
#define M 16

__global__ void matMult_largeBlocks(const float *A, const float *B, float *C, int n)
{	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	__shared__ float As[TILE_SIZE][TILE_SIZE];
	__shared__ float Bs[TILE_SIZE][TILE_SIZE];

	float Csub1 = 0.f;
	float Csub2 = 0.f;

	for (int b = 0; b < n/TILE_SIZE; ++b) {
		As[ty][tx] = A[(by * blockDim.y + ty) * n + (b * TILE_SIZE + tx)];
		As[ty+M][tx] = A[(by * blockDim.y + ty + M) * n + (b * TILE_SIZE + tx)];
		Bs[ty][tx] = B[(b * TILE_SIZE + ty) * n + (bx * blockDim.x + tx)];
		Bs[ty+M][tx] = B[(b * TILE_SIZE + ty + M) * n + (bx * blockDim.x + tx)];
		__syncthreads();

		#pragma unroll 32
		for (int k = 0; k < TILE_SIZE; ++k) {
			Csub1 += As[ty][k] * Bs[k][tx];
			Csub2 += As[ty+M][k] * Bs[k][tx];
		}
		__syncthreads();
	}

	C[(by * blockDim.y + ty) * n + (bx * blockDim.x + tx)] = Csub1;
	C[(by * blockDim.y + ty + M) * n + (bx * blockDim.x + tx)] = Csub2;
}

__global__ void matMult_tiled(const float *A, const float *B, float *C, int n)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	__shared__ float As[TILE_SIZE][TILE_SIZE];
	__shared__ float Bs[TILE_SIZE][TILE_SIZE];

	float Csub = 0.f;
	for (int b = 0; b < n/TILE_SIZE; b++) {
		As[ty][tx] = A[(by * blockDim.y + ty) * n + (b * TILE_SIZE + tx)];
		Bs[ty][tx] = B[(b * TILE_SIZE + ty) * n + (bx * blockDim.x + tx)];
		__syncthreads();

		for (int k = 0; k < TILE_SIZE; k++) {
			Csub += As[ty][k] * Bs[k][tx];
		}
		__syncthreads();
	}

	C[(by * blockDim.y + ty) * n + (bx * blockDim.x + tx)] = Csub;
}

__global__ void matMult_naive(const float *A, const float *B, float *C, int n)
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

//void printMatrix(const char *name, float *M, int n)
//{
//	printf("%s:\n", name);
//
//	int i, j;
//	for (i = 0; i < n; ++i) {
//		for (j = 0; j < n; ++j) {
//			printf("%8.2f ", M[i * n + j]);
//		}
//		printf("\n");
//	}
//}

void fillMatrices(float *A, float *B)
{	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			A[i * N + j] = 10.f * (float) rand() / (float) RAND_MAX;
			B[i * N + j] = 10.f * (float) rand() / (float) RAND_MAX;
		}
	}
}

int main(void)
{

#if defined LARGE_BLOCKS
	size_t blockSize = TILE_SIZE;
	dim3 threadsPerBlock(blockSize, M);
	dim3 blocksPerGrid(ceil(N/blockSize), ceil(N/blockSize));
#elif defined TILED
	size_t blockSize = TILE_SIZE;
	dim3 threadsPerBlock(blockSize, blockSize);
	dim3 blocksPerGrid(ceil(N/blockSize), ceil(N/blockSize));
#else // NAIVE
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(ceil(N/16), ceil(N/16));
#endif


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
	cudaMemcpy(dC, hC, matrixSizeBytes, cudaMemcpyHostToDevice);

	cudaEventRecord(start);
#if defined LARGE_BLOCKS
	matMult_largeBlocks<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, N);
#elif defined TILED
	matMult_tiled<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, N);
#else
	matMult_naive<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, N);
#endif

	cudaEventRecord(stop);

	cudaMemcpy(hC, dC, matrixSizeBytes, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

#if 0
	printMatrix("hA", hA, N);
	puts("+");
	printMatrix("hB", hB, N);
	puts("=");
	printMatrix("hC", hC, N);
#endif

#if defined LARGE_BLOCKS
	printf("Large blocks version\n");
#elif defined TILED
	printf("Tiled version\n");
#else
	printf("Naive version\n");	
#endif
	printf("N = %d\n", N);
	printf("Calculation status: %s\n", hC[0] != 0 ? "success" : "failed");
	printf("Threads per block: %u x %u = %u\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.x * threadsPerBlock.y);
	printf("Blocks per grid: %u x %u = %u\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.x * blocksPerGrid.y);
	printf("Elapsed time: %f ms\n", milliseconds);
	printf("GPU performance: %f megaevals/s\n", float(N*N)/milliseconds/1000.f);

	status = EXIT_SUCCESS;

cleanup:
	cudaFree(dA); cudaFree(dB); cudaFree(dC);
	free(hA); free(hB); free(hC);

	return status;
}
