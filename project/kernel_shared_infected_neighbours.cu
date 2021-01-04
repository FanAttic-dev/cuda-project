#define BLOCK_SIZE 16 // max 32
#define REDUCTION_BLOCK_SIZE 512

__device__ void warp_reduce(volatile int *shared_data, int tid)
{
	shared_data[tid] += shared_data[tid + 32];
	shared_data[tid] += shared_data[tid + 16];
	shared_data[tid] += shared_data[tid + 8];
	shared_data[tid] += shared_data[tid + 4];
	shared_data[tid] += shared_data[tid + 2];
	shared_data[tid] += shared_data[tid + 1];
}

__shared__ bool shared_infected_neighbours[BLOCK_SIZE][BLOCK_SIZE];

__device__ void calculate_infected_neighbours(const int* const in, const int n, const int x, const int y, const int* const contacts)
{
	int infected_neighbours = 0;
	for (int dy = -1; dy <= 1; ++dy) {
		for (int dx = -1; dx <= 1; ++dx) {
			// check bounds
			bool out_of_bounds = ((dx == 0 && dy == 0) || (x + dx < 0) || (x + dx >= n) || (y + dy < 0) || (y + dy >= n));

			infected_neighbours += !out_of_bounds && in[(y + dy) * n + x + dx] > 0 ? 1 : 0;
		}
	}

	shared_infected_neighbours[threadIdx.y][threadIdx.x] = infected_neighbours > contacts[y * n + x];
}

__global__ void make_iteration(const int* const contacts, const int* const in, const int n, const int iter, int* const out, int* const iter_block_infections)
{
	__shared__ int shared_iter_block_infections[BLOCK_SIZE * BLOCK_SIZE];
	const int idx_local = threadIdx.y * blockDim.x + threadIdx.x;
	shared_iter_block_infections[idx_local] = 0;

	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (y >= n || x >= n)
		return;

	const int idx_global = y * n + x;
	
	calculate_infected_neighbours(in, n, x, y, contacts);

	__syncthreads();

	const int house_in = in[idx_global];
	int house_out = 0;

	if (house_in == 0) { // healthy
	//	if ((threadIdx.x == 0 || threadIdx.x == BLOCK_SIZE-1) || (threadIdx.y == 0 || threadIdx.y == BLOCK_SIZE-1))
	//		infected = check_neighbours_global(in, n, x, y, goal);
	//	else
	//		infected = check_neighbours_shared(goal);
		// compare to connectivity
		if (shared_infected_neighbours[threadIdx.y][threadIdx.x]) {
			house_out = 10;
			++shared_iter_block_infections[idx_local];
		}
	} else if (house_in > 0) { // infected
		house_out = house_in - 1 == 0 ? -30 : house_in - 1;
	} else { // (house_in < 0) // recovering, immune
		house_out = house_in + 1;
	} 
	__syncthreads();


	// reduction
	// sum and save the total number of new infections in this iteration per block

	for (unsigned int s = BLOCK_SIZE * BLOCK_SIZE / 2; s > 32; s >>= 1) {
		if (idx_local < s) {
			shared_iter_block_infections[idx_local] += shared_iter_block_infections[idx_local + s];
		}
		__syncthreads();
	}

	if (idx_local < 32) {
		warp_reduce(shared_iter_block_infections, idx_local);
		__syncthreads();
	}

	if (idx_local == 0) {
		int iter_block_idx = (iter * gridDim.x * gridDim.y) + (blockIdx.y * gridDim.x + blockIdx.x);
		iter_block_infections[iter_block_idx] = shared_iter_block_infections[0];
	}

	out[idx_global] = house_out;
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

	for (int i = 1; i < blocks_per_iter; ++i) {
		shared[tid] += iter_block_infections[infections_idx + tid + (i * blockDim.x)];
	}

	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s)
			shared[tid] += shared[tid + s];

		__syncthreads();
	}

	if (tid < 32) {
		warp_reduce(shared, tid);
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
