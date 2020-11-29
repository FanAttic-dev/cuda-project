#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "kernel.cu"
#include "kernel_CPU.C"

#define INITIAL_SPREAD 0.05
#define CONTACTS_THRESHOLD 5

#define N /*2048*/ 1024
#define ITERS 100 //XXX more iterations may show some interesting effects, but are computationally demanding

#define DEBUG

void createInitialSpread(int *city, int n) {
    int sum = 0;
    for (int i = 0; i < n*n; i++) {
        if ((float)rand() / (float)RAND_MAX < INITIAL_SPREAD) {
            city[i] = 10.0f * (float)rand() / (float)RAND_MAX;
            sum++;
        }
        else
            city[i] = 0;
    }
    printf("Initially infected: %i\n", sum);
}

void generateContacts(int *contacts, int n) {
    for (int i = 0; i < n*n; i++)
        contacts[i] = lround((float)rand() / (float)RAND_MAX * (float)CONTACTS_THRESHOLD);
}

int main(int argc, char **argv){
    int *contacts = NULL;       // contact numbers for city
    int *city = NULL;           // homes computed by CPU
    int *infections = NULL;     // time serie of infections
    int *cityGPU = NULL;        // CPU buffer for GPU results
    int *infectionsGPU = NULL;  // CPU buffer for GPU results
    int *dContacts = NULL;      // GPU copy of contact numbers
    int *dCity = NULL;          // homes in GPU memory
    int *dInfections = NULL;    // time serie in GPU memory

    srand(123);

    // parse command line
    int device = 0;
    if (argc == 2) 
        device = atoi(argv[1]);
    if (cudaSetDevice(device) != cudaSuccess){
        fprintf(stderr, "Cannot set CUDA device!\n");
        exit(1);
    }
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Using device %d: \"%s\"\n", device, deviceProp.name);

    // create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate and set host memory
    contacts = (int*)malloc(N*N*sizeof(contacts[0]));
    city = (int*)malloc(N*N*sizeof(contacts[0]));
    infections = (int*)malloc(ITERS*sizeof(infections[0]));
    cityGPU = (int*)malloc(N*N*sizeof(cityGPU[0]));
    infectionsGPU = (int*)malloc(ITERS*sizeof(infectionsGPU[0]));
    createInitialSpread(city, N);
    generateContacts(contacts, N);
 
    // allocate and set device memory
    if (cudaMalloc((void**)&dContacts, N*N*sizeof(dContacts[0])) != cudaSuccess) {
        fprintf(stderr, "Device memory allocation error!\n");
        goto cleanup;
    }
    if (cudaMalloc((void**)&dCity, N*N*sizeof(dCity[0])) != cudaSuccess) {
        fprintf(stderr, "Device memory allocation error!\n");
        goto cleanup;
    }
    if (cudaMalloc((void**)&dInfections, ITERS*sizeof(dInfections[0])) != cudaSuccess) {
        fprintf(stderr, "Device memory allocation error!\n");
        goto cleanup;
    }
    cudaMemcpy(dContacts, contacts, N*N*sizeof(dContacts[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(dCity, city, N*N*sizeof(dCity[0]), cudaMemcpyHostToDevice);

    // solve on CPU
    printf("Solving on CPU...\n");
    cudaEventRecord(start, 0);
    solveCPU(contacts, city, infections, N, ITERS);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("CPU performance: %f megaevals/s\n",
                float(N*N)*float(ITERS)/time/1e3f);

    // dummy copy, just to awake GPU
    cudaMemcpy(cityGPU, dCity, N*N*sizeof(dCity[0]), cudaMemcpyDeviceToHost);

    // solve on GPU
    printf("Solving on GPU...\n");
    cudaEventRecord(start, 0);
    solveGPU(dContacts, dCity, dInfections, N, ITERS);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("GPU performance: %f megaevals/s\n",
                float(N*N)*float(ITERS)/time/1e3f);

#ifdef DEBUG
    // dump results
    {printf("Infections spread in time\n\n");
    int total = 0;
    for (int i = 0; i < ITERS; i++) {
        printf("Day %i: %i newly infected.\n", i, infections[i]);
        total += infections[i];
    }
    printf("Infected in total: %i\n", total);}
#endif

    // check GPU results
    cudaMemcpy(cityGPU, dCity, N*N*sizeof(dCity[0]), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            if (cityGPU[i*N + j] != city[i*N + j]){
                printf("Error detected in city at [%i, %i]: %i should be %i.\n", i, j, cityGPU[i*N + j], city[i*N + j]);
                goto cleanup; // exit after the first error
            }
        }
    cudaMemcpy(infectionsGPU, dInfections, ITERS*sizeof(dInfections[0]), cudaMemcpyDeviceToHost);
    for (int i = 0; i < ITERS; i++)
        if (infectionsGPU[i] != infections[i]) {
            printf("Error detected in infections at [%i]: %i should be %i.\n", i, infectionsGPU[i], infections[i]);
            goto cleanup; // exit after the first error
        }
    printf("Test OK.\n");

cleanup:
    cudaEventDestroy(start);
        cudaEventDestroy(stop);

    if (dCity) cudaFree(dCity);
    if (dContacts) cudaFree(dContacts);
    if (dInfections) cudaFree(dInfections);

    if (city) free(city);
    if (cityGPU) free(cityGPU);
    if (contacts) free(contacts);
    if (infections) free(infections);
    if (infectionsGPU) free(infectionsGPU);

    return 0;
}
