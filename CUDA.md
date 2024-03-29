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

![Selection_001](images/Selection_001.png)







## Info

- 32 threads in a warp for NVIDIA GPUs
- **max 1024 threads per block**
  - if 1D: `dim3 threadsPerBlock(1024, 1, 1)`
  - if 2D: `dim3 threadsPerBlock(512, 2, 1)`
  - if 3D: `dim3 threadsPerBlock(256, 2, 2)`
- SIMT reconvergence
  - at the end of divergent code, a point of reconvergence is set by the compiler
  - **can create deadlocks**
  - <img src="images/Selection_003.png" alt="Selection_003" style="zoom:50%;" />
- **all threads run** on the same multiprocessor
  - multiple blocks **can run** on one multiprocessor
- no **out of order execution** (as opposed to CPU)
  - but when a warp is waiting for data from memory, **another warp may be executed**
- **no** or limited **cache**



## Memory

![Selection_002](images/Selection_002.png)

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
- use **when all threads access the same constant**

### Texture memory

- read-only, cached, 2D locality
- high latency
- possible interpolation implemented in HW

### Data Cache

- `__restrict__`

  - shared HW with textures
    - high bandwidth but also higher latency
  - `__restrict__` and `ldg()`

### System RAM

- use`cudaMallocHost()` instead of `malloc()`
- free using `cudaFreeHost()`
- you can allocate `page-locked` memory areas, which can't be swapped (to reduce complications brought by virtual addressing)
  - `cudaHostAllocPortable`
  - `cudaHostAllocWriteCombined` flag turns off caching for CPU allocated memory
  - `cudaHostAllocMapped` flag sets host memory mapping in the device address space

### Page locked memory

- to tell the OS **not to page the mallocated block in RAM** to allow copying of the whole block at once to the GPU memory
- <img src="images/Selection_026.png" alt="Selection_026" style="zoom:50%;" />


## Global Memory Access Optimization

- global memory bandwidth is low relative to arithmetic performance of GPU

  - 400-600 cycles latency

- to maximize throughput, avoid bad parallel access pattern

- Fermi caches

  - L1: 256 B per row, 16 kB or 48 kB per multiprocessor
  - L2: 32 B per row, 768 kB in total
  
  

### Coalesced Memory Access

- **global memory is split into 64B** segments

  - two of these segments are aggregated into **128B** segments

- one memory transation can transfer **32, 64, or 128B** words

- **access the memory in larger blocks!**

- `x = A[id + offset]`

  #### One small transaction 

![Selection_006](images/Selection_006.png)

#### 	One big transation (but used only 50 % of data)

![Selection_007](images/Selection_007.png)

#### 	Two small transations (but used only 50 % of data)

![Selection_008](images/Selection_008.png)

#### 	Offset comparison 

![Selection_010](images/Selection_010.png)



### Interleaved memory access 

- `x = A[id * factor]` 
- the higher the factor, the lower the bandwidth

![Selection_009](images/Selection_009.png)



##### Overfetching

- loading data into cache and using **only a fraction of it**
- you can turn caching off in cases when caching hinders performance



### Partition Camping (only older GPUs)

- the use of just a certain subset of memory regions
- only old NVIDIA GPUs, but also the modern AMDs
- the memory is split into 256B regions

![Selection_011](images/Selection_011.png)

- if the thread blocks (TB) access the same partition, the memory transfer is serialized



## Shared Memory Access Optimization

- shared memory organized into **memory banks**, which can be accessed in parallel
  - shared memory banks are organized such that successive 32-bit words are assigned to successive banks and the bandwidth is 32 bits per bank per  clock cycle.
  - For devices of compute capability 2.0, the warp size is 32 threads and  the number of banks is also 32. A shared memory request for a warp is  not split as with devices of compute capability 1.x, meaning that bank  conflicts can occur between threads in the first half of a warp and  threads in the second half of the same warp.

![Selection_012](images/Selection_012.png)

#### 	No bank conflict

<img src="images/Selection_013.png" alt="Selection_013" style="zoom: 67%;" />

### Bank Conflict

- a bank is supposed to answer two requests at the same time

  #### n-way bank conflict 

<img src="images/Selection_014.png" alt="Selection_014" style="zoom:67%;" />

- no need for alignment

![Selection_015](images/Selection_015.png)

- with interleaved access, **the constant should be odd**

![Selection_016](images/Selection_016.png)

- broadcast available (OK if threads in warp access the same data)


## Synchronization

### Within block

- barrier - `__syncthreads()`
  - use **when using shared memory**
- beware of unused compute capability - **use more blocks** to hide latency

<img src="images/Selection_004.png" alt="Selection_004" style="zoom: 33%;" />

### Among blocks

- **no global barrier**
- if you want to synchronize all blocks, you need to stop the current kernel and run a new one
  - but OK because kernel launch has **negligible HW overhead** and **low SW overhead**
- for newer GPUs there is a way of synchronization in atomic operations to avoid false sharing



## Instructions Speed

- GPU is good for single-precision floating point operations (since it's designed for graphics)
  - newer GPUs support double precision, or half-precision
- GPU has 2 SFU cores for special operations
- **GPU single-precision implementation of functions**
  - `__sin()` - faster but less precise
- addition and multiplication very fast
  - FMAD - add and mult in a single intruction
  - **double precision are significantly slower**
- if n is power of 2, we can utilize
  - `i/n` is equivalent to `i>>log2(n)`
  - `i%n` is equivalent to `i&(n-1)`

- **loop unrolling**
  - increase the size of the loop body to parallelize on the level of instructions
- investigate assembly
  - `cuobjdump`
  - `decuda`
- double can be 32-times slower than float!



## Tips

- avoid code branching
- minimize CPU->GPU memory transfers
  - transfer large blocks at once
  - computations and memory transfers should be overlapped
- try not to use two shared memory loads in one calculation
  - if possible, get one operand from the shared memory and one from registers (by coallesed loading data from global memory)
- avoid shared memory bank conflicts
- avoid global memory partition camping
- use int8 instead of int if possible
- use float instead of double if possible
- when optimizing, go from coarser to finer optimizations
- beware of synchronization blocks
- use many threads to hide memory latencies
  - while some warp is waiting for memory, the scheduler can run other warp
- don't run too many threads due to power consumption
  - rather assign more work to one thread
- test if API calls are successful
- clear ouput arrays for debugging purposes
- beware of out-of-bounds accesses
  - no exception thrown

### Problem Choice
- accelerate code only if it is necessary (based on profiling)
- large enough
- sufficient number of flops to memory transfers (consider slow PCI-E)
- power consumption higher on GPUs
- parallelizable problem
- difficult to parallelize if:
  - threads in warp access random addresses in memory
  - thread in warp diverge

### Optimization
- start with coarser optimizations and only after that proceed to finer ones

1. PCI-E transfers
2. global memory access (bandwidth, latency)
3. access to other types of memory
4. divergence
5. parallelism configuration (block size, amount of serial work per thread)
6. instruction optimization