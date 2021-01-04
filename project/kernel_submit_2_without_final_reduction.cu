#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define REDUCTION_BLOCK_SIZE 256

__device__ bool check_neighbours_border(const int* const in, const int n, const int x, const int y, const int threshold)
{	
	int infected_neighbours = 0;
	for (int dy = -1; dy <= 1; ++dy) {
		for (int dx = -1; dx <= 1; ++dx) {
			// check bounds
			if ((dx == 0 && dy == 0) || (x + dx < 0) || (x + dx >= n) || (y + dy < 0) || (y + dy >= n))
				continue;

			if (in[(y + dy) * n + (x + dx)] > 0)
				++infected_neighbours;
		}
	}

	return infected_neighbours > threshold;
}

__device__ bool check_neighbours_inner(const int* const in, const int n, const int x, const int y, const int threshold)
{	
	int infected_neighbours = 0;

	for (int dy = -1; dy <= 1; ++dy) {
		int row = (y + dy) * n;
		infected_neighbours += in[row + (x - 1)] > 0 ? 1 : 0;
		infected_neighbours += in[row + x] > 0 ? 1 : 0;
		infected_neighbours += in[row + (x + 1)] > 0 ? 1 : 0;
	}

	return infected_neighbours > threshold;
}

__global__ void make_iteration(const int* const in, int* const out, const int* const contacts, int* const iter_block_infections, const int iter, const int n)
{
	__shared__ int shared_iter_block_infections[BLOCK_SIZE_X * BLOCK_SIZE_Y];
	const int idx_local = threadIdx.y * blockDim.x + threadIdx.x;
	shared_iter_block_infections[idx_local] = 0;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= n || y >= n)
		return;

	const int idx_global = y * n + x;
	const int house_in = in[idx_global];
	int house_out = 0;

	if (house_in == 0) { // healthy
		bool infected;	
		if (x == 0 || y == 0 || x == n-1 || y == n-1)
			infected = check_neighbours_border(in, n, x, y, contacts[idx_global]);
		else
			infected = check_neighbours_inner(in, n, x, y, contacts[idx_global]);

		if (infected) {
			house_out = 10;
			++shared_iter_block_infections[idx_local];
		}
	} else if (house_in > 0) { // infected
		house_out = house_in - 1 == 0 ? -30 : house_in - 1;
	} else { // (house_in < 0) // recovering, immune
		house_out = house_in + 1;
	}

	__syncthreads();

	// --- reduction ---

	// sum and save the total number of new infections in this iteration per block
	int infections = shared_iter_block_infections[idx_local];
	for (unsigned int offset = (BLOCK_SIZE_X * BLOCK_SIZE_Y) >> 1; offset > 32; offset >>= 1) {
		if (idx_local < offset)
			shared_iter_block_infections[idx_local] = infections = infections + shared_iter_block_infections[idx_local + offset];

		__syncthreads();
	}

	// reduce last warp
	if (idx_local < 32) {
		infections += shared_iter_block_infections[idx_local + 32];

		for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1)
			infections += __shfl_down_sync(0xffffffff, infections, offset);
	}

	if (idx_local == 0) {
		const unsigned int iter_block_idx = (iter * gridDim.x * gridDim.y) + (blockIdx.y * gridDim.x + blockIdx.x);
		iter_block_infections[iter_block_idx] = infections;
	}

	out[idx_global] = house_out;
}

/*
   For each iteration, sums infections per block to compute the number of new infections per iteration.
*/
__global__ void sum_infections(int* const infections, const int* const iter_block_infections, const int iters, const dim3 grid_size)
{
	extern __shared__ int shared_row[];
	const int tid = threadIdx.x;
	const int iter = blockIdx.x;
	const int iter_idx = iter * grid_size.x * grid_size.y;

	if (tid >= grid_size.x)
		return;

	int iter_sum = 0;
#if 1
	
	if (tid == 0) {
		for (int i = 0; i < grid_size.x * grid_size.y; ++i) {
			iter_sum += iter_block_infections[iter_idx + i];
		}
	}
	
#elif 0
	for (int row = 0; row < grid_size.y; ++row) {
		for (int col_stride = 0; col_stride < ceil(grid_size.x / (float) blockDim.x); ++col_stride) {	
			int row_sum = shared_row[tid] = iter_block_infections[iter_idx + (row * grid_size.x) + (col_stride * blockDim.x) + tid];
			__syncthreads();

			// reduction
			for (unsigned int offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
				if ((tid < offset) && (tid + offset < grid_size.x))
					shared_row[tid] = row_sum = row_sum + shared_row[tid + offset];

				__syncthreads();
			}

			if (tid < 32) {
				row_sum += shared_row[tid + 32];

				for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1)
					row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
			}
			__syncthreads();

			if (tid == 0)
				iter_sum += row_sum;
		}
	}
#endif
	if (tid == 0) {
		infections[iter] = iter_sum;
	}
}

void solveGPU(const int* const contacts, int* const city, int* const infections, const int n, const int iters)
{
	dim3 grid_size = dim3(ceil(n / (float) BLOCK_SIZE_X), ceil(n / (float) BLOCK_SIZE_Y));

	int *in = city;
	int *out;

	// 3D array storing infections per block per iteration
	int *iter_block_infections; // [iters][grid_size][grid_size] 

	if ((cudaMalloc((void**)&out, n * n * sizeof(int)) != cudaSuccess)
			|| (cudaMalloc((void**)&iter_block_infections, iters * grid_size.x * grid_size.y * sizeof(int)) != cudaSuccess)) {
		fprintf(stderr, "cudaMalloc failed\n");
		return;
	}

	dim3 threads_per_block = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 blocks_per_grid = dim3(grid_size.x, grid_size.y);
	for (int iter = 0; iter < iters; ++iter) {
		make_iteration<<<blocks_per_grid, threads_per_block>>>(in, out, contacts, iter_block_infections, iter, n);

		int *tmp = in;
		in = out;
		out = tmp;
	}

	// sum infections per iteration
	threads_per_block = REDUCTION_BLOCK_SIZE;
	blocks_per_grid = iters;
	sum_infections<<<blocks_per_grid, threads_per_block, grid_size.x>>>(infections, iter_block_infections, iters, grid_size);

	if (in != city) {
		cudaMemcpy(city, in, n*n*sizeof(int), cudaMemcpyDeviceToDevice);
		cudaFree(in);
	} else {
		cudaFree(out);
	}

	cudaFree(iter_block_infections);
}
