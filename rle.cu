#include <iostream>
#include <string>

__global__ void backWardMask(std::string * input, int * mask, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if(index == 0) {
        mask[0] = 1;
        return;
    }

    for(int i = index; i < n - 1; i+=stride) {
        mask[i] = input[i] == input[i - 1] ? 0 : 1;
    }
}

// NVIDIA's upsweep and downsweep prefix sum (prescan)
// TO-DO -> eliminate bank conflicts
__global__ void prescan(float *g_odata, float *g_idata, int n) {
    extern __shared__ float temp[];
    int thid = threadIdx.x; 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = 1; 
    temp[thid] = 0;

    if(index < n) {
        temp[thid] = g_idata[index];
        temp[thid+1] = g_idata[index+1];
    }

    for (int d = blockDim.x>>1; d > 0; d >>= 1) {   // build sum in place up the tree 
        __syncthreads();    
        
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }    
        offset *= 2; 
    } 

    if (thid == 0) { temp[blockDim.x - 1] = 0; } // clear the last element

    for (int d = 1; d < blockDim.x; d *= 2) { // traverse down tree & build scan     
        offset >>= 1;      
        __syncthreads();      
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;     
            int bi = offset*(2*thid+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi]; 
            temp[bi] += t;       
        } 
    }  
    __syncthreads();

    g_odata[index] = temp[thid];  
    g_odata[index+1] = temp[thid+1]; 
}

__global__ void compactKernel(int * scannedMask, int * compactedMask, int * totalRuns, int n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (index == 0) {
        compactedMask[0] = 0;
    }

    for (int i = index; i < n; i+=stride) {

        if (i == (n - 1)) {
            compactedMask[scannedMask[i]] = i + 1;
            *totalRuns = scannedMask[i];
        }

        if (scannedMask[i] != scannedMask[i - 1]) {
            compactedMask[scannedMask[i] - 1] = i;
        }
    }
}

__global__ void scatterKernel(int * compactedMask, int * totalRuns, int * in, int * symbolsOut, int * countsOut) {
    
    int n = *totalRuns;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i+=stride) {
        int a = compactedMask[i];
        int b = compactedMask[i + 1];

        symbolsOut[i] = in[a];
        countsOut[i] = b - a;
    }
}

int main(int argc, char ** argv) {

    char * in = argv[1];
    int input_size = sizeof(argv[1])/sizeof(argv[0]);
    int * mask = new int[input_size];
    int * scannedMask = new int[input_size];
    int * compactedMask = new int[input_size];
    int * totalRuns = new int;

    cudaMallocManaged(&in, input_size * sizeof(char));
    cudaMallocManaged(&mask, input_size * sizeof(int));
    cudaMallocManaged(&scannedMask, input_size * sizeof(int));

    int blockSize; 
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, prescan, 0, 0);
    int gridSize = (input_size + blockSize - 1) / blockSize;

    backWardMask<<<8, 512>>>(in, mask, input_size);
    cudaDeviceSynchronize();
    prescan<<<gridSize, blockSize>>>(scannedMask, mask, input_size);
    cudaDeviceSynchronize();
    // compactKernel<<<blocks, 512>>>();
    // cudaDeviceSynchronize();
    // scatterKernel<<<,>>>();
    // cudaDeviceSynchronize();

    std::cout << *in << std::endl;
    std::cout << *mask << std::endl; 
    std::cout << *scannedMask << std::endl; 

    delete in;
    delete [] mask;
    delete [] scannedMask;
}