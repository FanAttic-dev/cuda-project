
void solveCPU(const int* const contacts, int* const city, int* const infections, const int n, const int iters)
{
	int *in = city;
	int *out = (int*) malloc(n * n * sizeof(int));

	for (int iter = 0; iter < iters; ++iter) {
		int inf_new = 0;
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				int house_in = in[i * n + j];
				int *house_out = out[i * n + j];

				if (house_in > 0) {
					// infected
					house_in--;
					*house_out = house_in == 0 ? -30 : house_in;
				} else if (house_in < 0) {
					// recovering, immune
					*house_out = ++house_in;
				} else {
					// healthy

					// check neighbors
					int inf_neighbours = 0;
					for (int ii = max(0, i-1); ii <= min(i+1, n-1); ++ii)
					for (int jj = max(0, j-1); jj <= min(j+1, n-1); ++jj)
						if (in[ii * n + jj] > 0)
							++inf_neighbours;
					// compare to connectivity
					if (inf_neighbours > contacts[i * n + j]) {
						*house_out = 10;
						inf_new++;
					} else {
						*house_out = 0;
					}
				}
			}
		}

		int *tmp = in;
		in = out;
		out = tmp;

		infections[iter] = inf_new;
	}
	
	if (in != city) {
		memcpy(city, in, n * n * sizeof(int));
		free(in);
	} else {
		free(out);
	}
}
