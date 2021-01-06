#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8

__device__ bool check_neighbours_border(const int* const in, const int n, const int x, const int y, const int threshold)
{	
	int infected_neighbours = 0;
	for (int dy = -1; dy <= 1; ++dy) {
		#pragma unroll
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

	int row = (y - 1) * n;
	infected_neighbours += in[row + (x - 1)] > 0 ? 1 : 0;
	infected_neighbours += in[row + x] > 0 ? 1 : 0;
	infected_neighbours += in[row + (x + 1)] > 0 ? 1 : 0;

	row += n;
	infected_neighbours += in[row + (x - 1)] > 0 ? 1 : 0;
	infected_neighbours += in[row + (x + 1)] > 0 ? 1 : 0;

	row += n;
	infected_neighbours += in[row + (x - 1)] > 0 ? 1 : 0;
	infected_neighbours += in[row + x] > 0 ? 1 : 0;
	infected_neighbours += in[row + (x + 1)] > 0 ? 1 : 0;

	return infected_neighbours > threshold;
}

__global__ void make_iteration(const int* const in, int* const out, const int* const contacts, int* const infections, const int iter, const int n)
{
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
			atomicAdd(&infections[iter], 1);
		}
	} else if (house_in > 0) { // infected
		house_out = house_in - 1 == 0 ? -30 : house_in - 1;
	} else { // (house_in < 0) // recovering, immune
		house_out = house_in + 1;
	}

	out[idx_global] = house_out;
}

void solveGPU(const int* const contacts, int* const city, int* const infections, const int n, const int iters)
{
	int *in = city;
	int *out;

	if (cudaMalloc((void**)&out, n * n * sizeof(int)) != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed\n");
		return;
	}

	dim3 threads_per_block = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 grid_size = dim3(ceil(n / (float) BLOCK_SIZE_X), ceil(n / (float) BLOCK_SIZE_Y));
	dim3 blocks_per_grid = dim3(grid_size.x, grid_size.y);
	for (int iter = 0; iter < iters; ++iter) {
		make_iteration<<<blocks_per_grid, threads_per_block>>>(in, out, contacts, infections, iter, n);

		int *tmp = in;
		in = out;
		out = tmp;
	}

	if (in != city) {
		cudaMemcpy(city, in, n*n*sizeof(int), cudaMemcpyDeviceToDevice);
		cudaFree(in);
	} else {
		cudaFree(out);
	}
}
