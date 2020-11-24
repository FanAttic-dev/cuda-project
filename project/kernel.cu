// write your code into this file
// your kernels can be implemented directly here, or included
// function solveGPU is a device function: it can allocate memory, call CUDA kernels etc.

__global__ void make_iteration_single(const int* const contacts, const int* const in, int* const infections, const int n, const int iter, int* const out)
{
	int inf_new = 0;

	//int i = blockIdx.y * blockDim.y + threadIdx.y;
	//int j = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			int idx = i * n + j;

			int house_in = in[idx];
			int *house_out = &out[idx];

			if (house_in > 0) {
				// infected
				*house_out = --house_in == 0 ? -30 : house_in;
			} else if (house_in < 0) {
				// recovering, immune
				*house_out = ++house_in;
			} else {
				// healthy

				// check neighbours
				int inf_neighbours = 0;
				for (int ii = max(0, i-1); ii <= min(i+1, n-1); ++ii)
					for (int jj = max(0, j-1); jj <= min(j+1, n-1); ++jj)
						if (in[ii * n + jj] > 0)
							++inf_neighbours;

				// compare to connectivity
				if (inf_neighbours > contacts[i * n + j]) {
					*house_out = 10;
					++inf_new;
				} else {
					*house_out = 0;
				}
			}
		}
	}

	infections[iter] = inf_new;
}

__global__ void make_iteration(const int* const contacts, const int* const in, int* const infections, const int n, const int iter, int* const out)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	int idx = i * n + j;

	int house_in = in[idx];
	int *house_out = &out[idx];

	if (house_in > 0) {
		// infected
		*house_out = --house_in == 0 ? -30 : house_in;
	} else if (house_in < 0) {
		// recovering, immune
		*house_out = ++house_in;
	} else {
		// healthy

		// check neighbours
		int inf_neighbours = 0;
		for (int ii = max(0, i-1); ii <= min(i+1, n-1); ++ii)
			for (int jj = max(0, j-1); jj <= min(j+1, n-1); ++jj)
				if (in[ii * n + jj] > 0)
					++inf_neighbours;

		// compare to connectivity
		if (inf_neighbours > contacts[i * n + j]) {
			*house_out = 10;
			++infections[iter];
		} else {
			*house_out = 0;
		}
	}
	__syncthreads();
}

void solveGPU(const int* const contacts, int* const city, int* const infections, const int n, const int iters)
{
	int *in = city;
	int *out;

	if (cudaMalloc((void**)&out, n*n*sizeof(int)) != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed\n");
		return;
	}

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(1, 1);

	for (int iter = 0; iter < iters; ++iter) {
		//infections[iter] = 0;
		make_iteration_single<<<blocksPerGrid, threadsPerBlock>>>(contacts, in, infections, n, iter, out);

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
