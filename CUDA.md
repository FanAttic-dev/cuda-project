# Cuda

```c
threadIdx.{x, y, z} // thread pos within block

blockDim.{x, y, z} // block size

blockIdx.{x, y, z} // block pos within grid

gridDim.{x, y, z} // grid size
```

```c
__device__ // run and call on device (GPU) only

__global__ // run on device and called from host

__host__ // run and call on host only

__host__ and __device__ can be combined
```

![Selection_001](/home/atti/Documents/pv197_cuda/images/Selection_001.png)



## Info

- 32 threads in a warp
- SIMT reconvergence
  - at the end of divergent code, a point of reconvergence is set by the compiler
  - **can create deadlocks**
  - <img src="/home/atti/Documents/pv197_cuda/images/Selection_003.png" alt="Selection_003" style="zoom:50%;" />

- **all threads run** on the same multiprocessor
  - multiple blocks **can run** on one multiprocessor
- no **out of order execution** (as opposed to CPU)
  - but when a warp is waiting for data from memory, **another warp may be executed**
- **no** or limited **cache**



## Memory

![Selection_002](/home/atti/Documents/pv197_cuda/images/Selection_002.png)

### Registers

- **thread scoped**
- used for local variables in kernel and variables for intermediate reults

### Local memory

- **thread scoped**
- for data **that doesn't fit into the registers**
- stored in DRAM => slow, high latency

### Shared memory

- **block scoped**

- `__shared__`

- a variable in shared memory can have dynamic size (determined at startup), if declared as `extern` without size specification

- ```c
  // --- static allocation
  __shared__ float myArray[128];
  
  // --- dynamic allocation
  // call the kernel with specified size
  myKernel<<<gird, block, n>>>();
  
  // in kernel
  extern __shared__ char myArray[];
  ```

### Global memory

- **application scoped**
- `__global__`
- access it in a **coalesced way**
- cached in some architectures (but only L2, not too fast)
- either dynamically allocated `cudaMalloc`
- or statically `__device__ int myVariable;`

### Constant memory

- **application scoped**
- `__constant__`
- **read-only**
- **cached**
  - cache hit as fast as registry
  - cache miss as fast as global memory
- limited size (64 kB)

### Data Cache

- `__restrict__`

### System RAM

- use`cudaMallocHost()` instead of `malloc()`
- free using `cudaFreeHost()`
- you can allocate `page-locked` memory areas, which can't be swapped (to reduce complications brought by virtual addressing)
  - `cudaHostAllocPortable`
  - `cudaHostAllocWriteCombined` flag turns off caching for CPU allocated memory
  - `cudaHostAllocMapped` flag sets host memory mapping in the device address space



## Synchronization

### Within block

- barrier - `__syncthreads()`
  - use **when using shared memory**
- beware of unused compute capability - **use more blocks** to hide latency

<img src="/home/atti/Documents/pv197_cuda/images/Selection_004.png" alt="Selection_004" style="zoom: 33%;" />

### Among blocks

- **no global barrier**
- if you want to synchronize all blocks, you need to stop the current kernel and run a new one
- for newer GPUs there is a way of synchronization in atomic operations to avoid false sharing





## Tips

- avoid code branching
- 