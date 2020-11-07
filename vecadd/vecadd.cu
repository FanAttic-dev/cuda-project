#include <stdio.h>

#define N 64
#define BLOCK 32

__global__ void addvec(float *a, float *b, float *c, int n) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		c[i] = a[i] + b[i];
}

void printvec(float *vec, int n)
{
	printf("(");
	for (int i = 0; i < n; ++i) {
		printf("%f", vec[i]);
		if (i < n - 1)
			printf(", ");
	}
	printf(")\n\n");
}

int main()
{
	float *a, *b, *c;

	cudaMallocManaged(&a, N*sizeof(*a));
	cudaMallocManaged(&b, N*sizeof(*b));
	cudaMallocManaged(&c, N*sizeof(*c));

	for (int i = 0; i < N; ++i) {
		a[i] = i;
		b[i] = i*3;
		c[i] = 0;
	}

	printvec(a, N);
	printvec(b, N);

	addvec<<<N/BLOCK + 1, BLOCK>>>(a, b, c, N);	

	cudaDeviceSynchronize();

	printvec(c, N);

	cudaFree(a); cudaFree(b); cudaFree(c);

	return 0;
}

int main_manual()
{
	size_t bytes = N*sizeof(float);

	// allocate memory
	float *h_a = (float*) malloc(bytes);
	float *h_b = (float*) malloc(bytes);
	float *h_c = (float*) malloc(bytes);
	
	float *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// fill values
	for (int i = 0; i < N; ++i) {
		h_a[i] = i;
		h_b[i] = i*3;
		h_c[i] = 0;
	}

	// CPU -> GPU
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	// print vectors
	printvec(h_a, N);
	printf("+\n\n");
	printvec(h_b, N);

	// compute
	size_t grid_size = N / BLOCK + 1;
	addvec<<<grid_size, BLOCK>>>(d_a, d_b, d_c, N);

	// GPU -> CPU
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	// print result
	printf("=\n\n");
	printvec(h_c, N);

	// release GPU memory
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	// release CPU memory
	free(h_a); free(h_b); free(h_c);

	return 0;
}
