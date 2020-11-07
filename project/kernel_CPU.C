// naive CPU implementation

void solveCPU(const int* const contacts, int* const city, int* const infections, const int n, const int iters) {
    int *in = city;
    int *out = (int*)malloc(n*n*sizeof(out[0]));

    for (int it = 0; it < iters; it++) {
        int infSum = 0;
        for (int i = 0; i < n; i++) 
            for (int j = 0; j < n; j++) {
                if (in[i*n + j] > 0) {
                    out[i*n + j] = in[i*n + j] - 1;
                    if (out[i*n + j] == 0)
                        out[i*n + j] = -30;
                }
                if (in[i*n + j] < 0) {
                    out[i*n + j] = in[i*n + j] + 1;
                }
                if (in[i*n + j] == 0) {
                    int infected = 0;
                    for (int ii = max(i-1, 0); ii <= min(i+1, n-1); ii++)
                    for (int jj = max(j-1, 0); jj <= min(j+1, n-1); jj++)
                        infected += (in[ii*n + jj] > 0) ? 1 : 0;
                    infected -= (in[i*n + j] > 0) ? 1 : 0;
                    if (infected > contacts[i*n + j]) {
                        out[i*n + j] = 10;
                        infSum++;
                    }
                    else
                        out[i*n + j] = 0;
                }
            }

        // flip in x out
        int *tmp = in;
        in = out;
        out = tmp;
	infections[it] = infSum;
    }

    if (in != city) {
        memcpy(city, in, n*n*sizeof(city[0]));
        free(in);
    }
    else
        free(out);
}

