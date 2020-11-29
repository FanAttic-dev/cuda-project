
//#define IN_SHARED
#define BLOCK_SIZE 8

__global__ void make_iteration_in_shared(const int* const contacts, const int* const in, const int n, const int iter, int* const out, int* const iter_block_infections)
{
	__shared__ int shared_iter_block_infections[BLOCK_SIZE][BLOCK_SIZE];
	shared_iter_block_infections[threadIdx.y][threadIdx.x] = 0;

	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (y >= n || x >= n)
		return;

	int idx = y * n + x;

	__shared__ int shared_in[BLOCK_SIZE][BLOCK_SIZE];
	int house_in = shared_in[threadIdx.y][threadIdx.x] = in[idx];

	__syncthreads();

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
			if ((x + dx < 0) || (x + dx >= n) || (y + dy < 0) || (y + dy >= n))
				continue;

			// block local coordinates
			int xx = threadIdx.x + dx;
			int yy = threadIdx.y + dy;

			int neighbor;
			// use shared memory if sampling within block
			if ((xx >= 0) && (xx < blockDim.x) && (yy >= 0) && (yy < blockDim.y))
				neighbor = shared_in[yy][xx];
			else // use global memory if out of block bounds
				neighbor = in[(y + dy) * n + (x + dx)];

			if (neighbor > 0)
				++inf_neighbours;
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

	// sum and save the total number of new infections in this iteration per block
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		int iter_block_idx = (iter * gridDim.x * gridDim.y) + (blockIdx.y * gridDim.x + blockIdx.x);
		iter_block_infections[iter_block_idx] = 0;
		for (int yy = 0; yy < BLOCK_SIZE; ++yy)
		for (int xx = 0; xx < BLOCK_SIZE; ++xx) {
			iter_block_infections[iter_block_idx] += shared_iter_block_infections[yy][xx];
		}
	}
}

__global__ void make_iteration_in_global(const int* const contacts, const int* const in, const int n, const int iter, int* const out, int* const iter_block_infections)
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
			if ((x + dx < 0) || (x + dx >= n) || (y + dy < 0) || (y + dy >= n))
				continue;

			int neighbor = in[(y + dy) * n + (x + dx)];

			if (neighbor > 0)
				++inf_neighbours;
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

	// sum and save the total number of new infections in this iteration per block
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		int iter_block_idx = (iter * gridDim.x * gridDim.y) + (blockIdx.y * gridDim.x + blockIdx.x);
		iter_block_infections[iter_block_idx] = 0;
		for (int yy = 0; yy < BLOCK_SIZE; ++yy)
		for (int xx = 0; xx < BLOCK_SIZE; ++xx) {
			iter_block_infections[iter_block_idx] += shared_iter_block_infections[yy][xx];
		}
	}
}

/*
   For each iteration, sums infections per block to compute the number of new infections per iteration.
*/
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
	int grid_size = ceil(n / (float) BLOCK_SIZE);

	int *in = city;
	int *out;
	int *iter_block_infections; 	// [iters][grid_size][grid_size] 
					// 3D array storing infections per block per iteration

	if (cudaMalloc((void**)&out, n * n * sizeof(int)) != cudaSuccess
			|| cudaMalloc((void**)&iter_block_infections, iters * grid_size * grid_size * sizeof(int)) != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed\n");
		return;
	}

	dim3 threads_per_block = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks_per_grid = dim3(grid_size, grid_size);
	for (int iter = 0; iter < iters; ++iter) {
#ifdef IN_SHARED
		make_iteration_in_shared<<<blocks_per_grid, threads_per_block>>>(contacts, in, n, iter, out, iter_block_infections);
#else
		make_iteration_in_global<<<blocks_per_grid, threads_per_block>>>(contacts, in, n, iter, out, iter_block_infections);
#endif
		int *tmp = in;
		in = out;
		out = tmp;
	}

	// reduce infections per iter, which are stored per block
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
