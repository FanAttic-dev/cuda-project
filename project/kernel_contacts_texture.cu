#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16
#define REDUCTION_BLOCK_SIZE 512

texture<int, 2, cudaReadModeElementType> tex_contacts;

__device__ void warp_reduce(volatile int *shared_data, int tid)
{
	shared_data[tid] += shared_data[tid + 32];
	shared_data[tid] += shared_data[tid + 16];
	shared_data[tid] += shared_data[tid + 8];
	shared_data[tid] += shared_data[tid + 4];
	shared_data[tid] += shared_data[tid + 2];
	shared_data[tid] += shared_data[tid + 1];
}

__device__ bool check_neighbours(const int* const city, const int n, const int x, const int y, const int threshold)
{
	// check neighbours
	int inf_neighbours = 0;
	for (int dy = -1; dy <= 1; ++dy) {
		for (int dx = -1; dx <= 1; ++dx) { 
			if (city[(y + dy) * n + (x + dx)] > 0)
				++inf_neighbours;
		}
	}

	return inf_neighbours > threshold;
}

__global__ void make_iteration(const int* const contacts, const int* const in, int* const out, const int n, const int iter, int* const iter_block_infections)
{
	__shared__ int shared_iter_block_infections[BLOCK_HEIGHT * BLOCK_WIDTH];
	const int idx_local = threadIdx.y * blockDim.x + threadIdx.x;
	shared_iter_block_infections[idx_local] = 0;

	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (y >= n || x >= n)
		return;

	const int idx_global = y * n + x;

	const int house_in = in[idx_global];
	int house_out = 0;

	if (house_in == 0) { // healthy
		const int threshold = tex2D(tex_contacts, x, y);
		bool infected = check_neighbours(in, n, x, y, threshold);

		// compare to connectivity
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

/*	
	// sum and save the total number of new infections in this iteration per block
	if (idx_local == 0) {
		int iter_block_idx = (iter * gridDim.x * gridDim.y) + (blockIdx.y * gridDim.x + blockIdx.x);
		iter_block_infections[iter_block_idx] = 0;
		for (int yy = 0; yy < BLOCK_SIZE; ++yy)
		for (int xx = 0; xx < BLOCK_SIZE; ++xx)
			iter_block_infections[iter_block_idx] += shared_iter_block_infections[yy * BLOCK_SIZE + xx];
	}
*/

	// reduction
	// sum and save the total number of new infections in this iteration per block

	for (unsigned int s = BLOCK_HEIGHT * BLOCK_WIDTH / 2; s > 32; s >>= 1) {
		if (idx_local < s) {
			shared_iter_block_infections[idx_local] += shared_iter_block_infections[idx_local + s];
		}
		__syncthreads();
	}
/*
	// TODO make generic
	if (idx_local < 128) {
		shared_iter_block_infections[idx_local] += shared_iter_block_infections[idx_local + 128];
		__syncthreads();
	}

	if (idx_local < 64) {
		shared_iter_block_infections[idx_local] += shared_iter_block_infections[idx_local + 64];
		__syncthreads();
	}
*/

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
__global__ void reduce_infections(int* const infections, const int* const iter_block_infections, const int iters, const int blocks_per_iter, const dim3 grid_size)
{
	__shared__ int shared[REDUCTION_BLOCK_SIZE];

	const int tid = threadIdx.x;
	const int iter = blockIdx.x;
	const int infections_idx = iter * grid_size.x * grid_size.y;
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
	size_t size = n * n * sizeof(int);
	dim3 threads_per_block = dim3(BLOCK_WIDTH, BLOCK_HEIGHT);
	dim3 grid_size = dim3(ceil(n / (float) threads_per_block.x), ceil(n / (float) threads_per_block.y));

	int *in = city;
	int *out;
	int *iter_block_infections; 	// [iters][grid_size][grid_size] 
					// 3D array storing infections per block per iteration
	cudaArray *tex_array_contacts;
	cudaChannelFormatDesc channel_desc = 
		cudaCreateChannelDesc(sizeof(int) * 8, 0, 0, 0, cudaChannelFormatKindSigned);
	if ((cudaMalloc((void**)&out, size) != cudaSuccess)
			|| (cudaMallocArray(&tex_array_contacts, &channel_desc, n, n) != cudaSuccess)
		       	|| (cudaMalloc((void**)&iter_block_infections, iters * grid_size.x * grid_size.y * sizeof(int)) != cudaSuccess)) {
		fprintf(stderr, "cudaMalloc failed\n");
		return;
	}

	tex_contacts.addressMode[0] = cudaAddressModeBorder;
	tex_contacts.addressMode[1] = cudaAddressModeBorder;
	tex_contacts.filterMode = cudaFilterModePoint;
	tex_contacts.normalized = false;

	if (cudaMemcpyToArray(tex_array_contacts, 0, 0, contacts, size, cudaMemcpyDeviceToDevice) != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToArray failed\n");
		return;
	}

	if (cudaBindTextureToArray(tex_contacts, tex_array_contacts, channel_desc) != cudaSuccess) {
		fprintf(stderr, "cudaBindTextureToArray failed\n");
		return;
	}

	dim3 blocks_per_grid = grid_size;
	for (int iter = 0; iter < iters; ++iter) {
		make_iteration<<<blocks_per_grid, threads_per_block>>>(contacts, in, out, n, iter, iter_block_infections);

		int* tmp = in;
		in = out;
		out = tmp;
	}

	// reduce infections per iter, which are stored per block
	threads_per_block = REDUCTION_BLOCK_SIZE;
	blocks_per_grid = iters;
	int blocks_per_iter = (grid_size.x * grid_size.y) / REDUCTION_BLOCK_SIZE;
	reduce_infections<<<blocks_per_grid, threads_per_block>>>(infections, iter_block_infections, iters, blocks_per_iter, grid_size);

	if (in != city) {
		cudaMemcpy(city, in, size, cudaMemcpyDeviceToDevice);
		cudaFree(in);
	} else {
		cudaFree(out);
	}

	cudaFreeArray(tex_array_contacts);
	cudaFree(iter_block_infections);
}
