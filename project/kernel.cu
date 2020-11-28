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
				if (inf_neighbours > contacts[idx]) {
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

__global__ void make_iteration(const int* const contacts, const int* const in, int* const infections, const int n, const int iter, int* const out, int* const iter_infections)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	int idx = i * n + j;

	int house_in = in[idx];
	int house_out = -1;
	iter_infections[idx] = 0;

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
			iter_infections[idx] = 1;
		} else {
			house_out = 0;
		}
	}
	out[idx] = house_out;

/*
	if (idx == 0) {
		int sum = 0;
		for (int ii = 0; ii < n*n; ++ii) {
			sum += iter_infections[ii];
		}
		
		infections[iter] = sum;
	}
*/
}

void solveGPU(const int* const contacts, int* const city, int* const infections, const int n, const int iters)
{
	int *in = city;
	int *out;
	int *iter_infections;

	if (cudaMalloc((void**)&out, n*n*sizeof(int)) != cudaSuccess
			|| cudaMalloc((void**)&iter_infections, n*n*sizeof(int)) != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed\n");
		return;
	}

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(n/16, n/16);

	for (int iter = 0; iter < iters; ++iter) {
		make_iteration<<<blocksPerGrid, threadsPerBlock>>>(contacts, in, infections, n, iter, out, iter_infections);

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

	cudaFree(iter_infections);
}
