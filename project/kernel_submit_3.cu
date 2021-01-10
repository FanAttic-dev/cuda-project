#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 2

__global__ void make_iteration_inner(const int* const in, int* const out, const int* const contacts, int* const infections, const int iter, const int n)
{
	const int x = (blockIdx.x+1) * blockDim.x + threadIdx.x;
	const int y = (blockIdx.y+1) * blockDim.y + threadIdx.y;

	const int idx_global = y * n + x;

	const int house_in = in[idx_global];	
	int house_out = 0;

	if (house_in == 0) { // healthy
		int infected_neighbours = 0;
		for (int dy = -1; dy <= 1; ++dy) {
			int row = (y + dy) * n;
			infected_neighbours += in[row + x - 1] > 0 ? 1 : 0;
			infected_neighbours += in[row + x] > 0 ? 1 : 0;
			infected_neighbours += in[row + x + 1] > 0 ? 1 : 0;
		}

		if (infected_neighbours > contacts[idx_global]) {
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

__global__ void make_iteration_border_x(const int* const in, int* const out, const int* const contacts, int* const infections, const int iter, const int n, dim3 grid_size)
{
	const int x = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
	const int y = (blockIdx.y * (grid_size.y-1)) * blockDim.y + threadIdx.y;

	const int idx_global = y * n + x;

	const int house_in = in[idx_global];
	int house_out = 0;

	if (house_in == 0) { // healthy
		int infected_neighbours = 0;
		for (int dy = -1; dy <= 1; ++dy) {
			if ((y + dy < 0) || (y + dy >= n))
				continue;

			const int row = (y + dy) * n;
			infected_neighbours += in[row + x - 1] > 0 ? 1 : 0;
			infected_neighbours += in[row + x] > 0 ? 1 : 0;
			infected_neighbours += in[row + x + 1] > 0 ? 1 : 0;
		}

		if (infected_neighbours > contacts[idx_global]) {
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

__global__ void make_iteration_border_y(const int* const in, int* const out, const int* const contacts, int* const infections, const int iter, const int n, dim3 grid_size)
{
	const int x = (blockIdx.x * (grid_size.x-1)) * blockDim.x + threadIdx.x;
	const int y = (blockIdx.y + 1) * blockDim.y + threadIdx.y;

	const int idx_global = y * n + x;

	const int house_in = in[idx_global];
	int house_out = 0;

	if (house_in == 0) { // healthy
		int infected_neighbours = 0;
		for (int dx = -1; dx <= 1; ++dx) {
			// check bounds
			if ((x + dx < 0) || (x + dx >= n))
				continue;

			infected_neighbours += in[(y - 1) * n + x + dx] > 0 ? 1 : 0;
			infected_neighbours += in[(y) * n + x + dx] > 0 ? 1 : 0;
			infected_neighbours += in[(y + 1) * n + x + dx] > 0 ? 1 : 0;
			
		}

		if (infected_neighbours > contacts[idx_global]) {
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

__global__ void make_iteration_corner(const int* const in, int* const out, const int* const contacts, int* const infections, const int iter, const int n, dim3 grid_size)
{
	const int x = (blockIdx.x * (grid_size.x-1)) * blockDim.x + threadIdx.x;
	const int y = (blockIdx.y * (grid_size.y-1)) * blockDim.y + threadIdx.y;

	const int idx_global = y * n + x;

	const int house_in = in[idx_global];
	int house_out = 0;

	if (house_in == 0) { // healthy
		int infected_neighbours = 0;
		for (int dy = -1; dy <= 1; ++dy) {
			const int row = (y + dy) * n;
			for (int dx = -1; dx <= 1; ++dx) {
				// check bounds
				if ((x + dx >= 0) && (x + dx < n) && (y + dy >= 0) && (y + dy < n) && (in[row + (x + dx)] > 0))
					++infected_neighbours;
			}
		}

		if (infected_neighbours > contacts[idx_global]) {
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
	for (int iter = 0; iter < iters; ++iter) {
		make_iteration_inner<<<dim3(grid_size.x-2, grid_size.y-2), threads_per_block>>>(in, out, contacts, infections, iter, n);
		make_iteration_corner<<<dim3(2, 2), threads_per_block>>>(in, out, contacts, infections, iter, n, grid_size);
		make_iteration_border_y<<<dim3(2, grid_size.y-2), threads_per_block>>>(in, out, contacts, infections, iter, n, grid_size);
		make_iteration_border_x<<<dim3(grid_size.x-2, 2), threads_per_block>>>(in, out, contacts, infections, iter, n, grid_size);

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
