// write your code into this file
// your kernels can be implemented directly here, or included
// function solveGPU is a device function: it can allocate memory, call CUDA kernels etc.

#define BLOCK_SIZE 8

__global__ void make_iteration(const int* const contacts, const int* const in, const int n, const int iter, int* const out, int* const iter_block_infections)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	int idx = i * n + j;

	if (idx >= n * n)
		return;

	__shared__ int shared_iter_block_infections[BLOCK_SIZE][BLOCK_SIZE];
	shared_iter_block_infections[threadIdx.y][threadIdx.x] = 0;

	int house_in = in[idx];
	int house_out = -1;

	if (house_in > 0) {
		// infected
		house_out = --house_in == 0 ? -30 : house_in;
	} else if (house_in < 0) {
		// recovering, immune
		house_out = ++house_in;
	} else {
		// healthy

		// check neighbours
		int inf_neighbours = 0;
		for (int ii = max(0, i-1); ii <= min(i+1, n-1); ++ii)
		for (int jj = max(0, j-1); jj <= min(j+1, n-1); ++jj)
			if (in[ii * n + jj] > 0)
				++inf_neighbours;

		// compare to connectivity
		if (inf_neighbours > contacts[idx]) {
			house_out = 10;
			++shared_iter_block_infections[threadIdx.y][threadIdx.x];
		} else {
			house_out = 0;
		}
	}
	out[idx] = house_out;

	__syncthreads();

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		iter_block_infections[(iter * gridDim.x * gridDim.y) + (blockIdx.y * gridDim.x + blockIdx.x)] = 0;
		for (int ii = 0; ii < BLOCK_SIZE; ++ii)
		for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
			iter_block_infections[(iter * gridDim.x * gridDim.y) + (blockIdx.y * gridDim.x + blockIdx.x)] += shared_iter_block_infections[ii][jj];
		}
	}
}

__global__ void reduce_infections(int* const infections, const int* const iter_block_infections, const int iters, const int grid_size)
{
	int iter = blockIdx.x * blockDim.x + threadIdx.x;

	if (iter >= iters)
		return;

	infections[iter] = 0;

	for (int ii = 0; ii < grid_size; ++ii)
	for (int jj = 0; jj < grid_size; ++jj)
		infections[iter] += iter_block_infections[(iter * grid_size * grid_size) + (ii * grid_size + jj)];
}

void solveGPU(const int* const contacts, int* const city, int* const infections, const int n, const int iters)
{
	int *in = city;
	int *out;
	int *iter_block_infections; // 3D array storing infections per block per iteration

	int grid_size = ceil(n / (float) BLOCK_SIZE);
	printf("Block count: %d\n", grid_size);

	if (cudaMalloc((void**)&out, n*n*sizeof(int)) != cudaSuccess
			|| cudaMalloc((void**)&iter_block_infections, iters * grid_size * grid_size * sizeof(int)) != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed\n");
		return;
	}

	dim3 threads_per_block = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks_per_grid = dim3(grid_size, grid_size);
	for (int iter = 0; iter < iters; ++iter) {
		make_iteration<<<blocks_per_grid, threads_per_block>>>(contacts, in, n, iter, out, iter_block_infections);

		int *tmp = in;
		in = out;
		out = tmp;
	}

	threads_per_block = 32;
	blocks_per_grid = ceil(iters / (float) threads_per_block.x);
	reduce_infections<<<blocks_per_grid, threads_per_block>>>(infections, iter_block_infections, iters, grid_size);

	if (in != city) {
		cudaMemcpy(city, in, n*n*sizeof(int), cudaMemcpyDeviceToDevice);
		cudaFree(in);
	} else {
		cudaFree(out);
	}

	cudaFree(iter_block_infections);
}
