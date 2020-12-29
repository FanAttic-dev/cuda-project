#define BLOCK_SIZE 32
#define REDUCTION_BLOCK_SIZE 1024

__global__ void make_iteration(const int* const contacts, const int* const in, const int n, const int iter, int* const out, int* const iter_block_infections)
{
	__shared__ int shared_iter_block_infections[BLOCK_SIZE][BLOCK_SIZE];
	shared_iter_block_infections[threadIdx.y][threadIdx.x] = 0;

	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (y >= n || x >= n)
		return;

	int idx = y * n + x;

	int house_in = in[idx];
	int house_out;

	if (house_in > 0) { // infected
		house_out = --house_in == 0 ? -30 : house_in;
	} else if (house_in < 0) { // recovering, immune
		house_out = ++house_in;
	} else { // healthy
		// check neighbours
		int inf_neighbours = 0;
		for (int dy = -1; dy <= 1; ++dy)
		for (int dx = -1; dx <= 1; ++dx) {
			// check bounds
			if ((x + dx < 0) || (x + dx >= n) || (y + dy < 0) || (y + dy >= n) || (dx == 0 && dy == 0))
				continue;

			inf_neighbours += (in[(y + dy) * n + (x + dx)] > 0) ? 1 : 0;
		}

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

	// reduction
	// sum and save the total number of new infections in this iteration per block

	// BLOCK_SIZE / 2 x BLOCK_SIZE / 2
	int reduction_block_size = BLOCK_SIZE / 2;
	if ((threadIdx.x % reduction_block_size == 0) && (threadIdx.y % reduction_block_size == 0)) {
		for (int yy = 0; yy < reduction_block_size; ++yy) {
		for (int xx = 0; xx < reduction_block_size; ++xx) {
			if (xx + yy == 0)
				continue;
			shared_iter_block_infections[threadIdx.y][threadIdx.x] += 
				shared_iter_block_infections[threadIdx.y + yy][threadIdx.x + xx];
		}
		}
	}

	__syncthreads();

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		int iter_block_idx = (iter * gridDim.x * gridDim.y) + (blockIdx.y * gridDim.x + blockIdx.x);
		iter_block_infections[iter_block_idx] = 0;
		for (int yy = 0; yy < BLOCK_SIZE; yy += reduction_block_size)
		for (int xx = 0; xx < BLOCK_SIZE; xx += reduction_block_size)
			iter_block_infections[iter_block_idx] += shared_iter_block_infections[yy][xx];
	}
}

/*
   For each iteration, sums infections per block to compute the number of new infections per iteration.
*/
__global__ void reduce_infections(int* const infections, const int* const iter_block_infections, const int iters, const int blocks_per_iter, const int grid_size)
{
	__shared__ int shared[REDUCTION_BLOCK_SIZE];

	const int tid = threadIdx.x;
	const int iter = blockIdx.x;
	const int infections_idx = iter * grid_size * grid_size;
	shared[tid] = iter_block_infections[infections_idx + tid];

	// 2048 / 1024
	for (int i = 1; i < blocks_per_iter; ++i) {
		shared[tid] += iter_block_infections[infections_idx + tid + (i * blockDim.x)];
	}

	__syncthreads();

	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			shared[tid] += shared[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		infections[iter] = shared[0];
	}
}

void solveGPU(const int* const contacts, int* const city, int* const infections, const int n, const int iters)
{
	int grid_size = ceil(n / (float) BLOCK_SIZE);

	int *in = city;
	int *out;
	int *iter_block_infections; 	// [iters][grid_size][grid_size] 
					// 3D array storing infections per block per iteration

	if ((cudaMalloc((void**)&out, n * n * sizeof(int)) != cudaSuccess)
			|| (cudaMalloc((void**)&iter_block_infections, iters * grid_size * grid_size * sizeof(int)) != cudaSuccess)) {
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

	// reduce infections per iter, which are stored per block
	threads_per_block = REDUCTION_BLOCK_SIZE;
	blocks_per_grid = iters;
	int blocks_per_iter = (grid_size * grid_size) / REDUCTION_BLOCK_SIZE;
	reduce_infections<<<blocks_per_grid, threads_per_block>>>(infections, iter_block_infections, iters, blocks_per_iter, grid_size);

	if (in != city) {
		cudaMemcpy(city, in, n*n*sizeof(int), cudaMemcpyDeviceToDevice);
		cudaFree(in);
	} else {
		cudaFree(out);
	}

	cudaFree(iter_block_infections);
}
