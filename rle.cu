#include <iostream>
#include <string>
#include <math.h>
#include <stdio.h>

__global__ void backWardMask(char * input, int * mask, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if(index == 0) {
        mask[0] = 1;
        for(int i = index + stride; i < n; i+=stride) {
            mask[i] = input[i] == input[i - 1] ? 0 : 1;
        }
        return;
    }

    for(int i = index; i < n - 1; i+=stride) {
        mask[i] = input[i] == input[i - 1] ? 0 : 1;
    }
}

// NVIDIA's upsweep and downsweep prefix sum (prescan)
// TO-DO -> eliminate bank conflicts
__global__ void prescan(int *g_odata, int *g_idata, int * blockSums, int n) {
    extern __shared__ int temp[];
    int thid = threadIdx.x; 
    int index = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    int offset = 1; 

    temp[2*thid] = 0;
    temp[2*thid+1] = 0;

    if(index < n) {
        temp[2*thid] = g_idata[index];
        temp[2*thid+1] = g_idata[index+1];
    }

    // build sum in place up the tree 
    for (int d = 2*blockDim.x>>1; d > 0; d >>= 1) {
        __syncthreads();    
        
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }    
        offset *= 2;
    } 

    // clear the last element
    if (thid == 0) {
        blockSums[blockIdx.x] = temp[2*blockDim.x - 1];
        temp[2*blockDim.x - 1] = 0; 
    }

    // traverse down tree & build scan  
    for (int d = 1; d < 2*blockDim.x; d *= 2) {   
        offset >>= 1;      
        __syncthreads();      
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;     
            int bi = offset*(2*thid+2)-1;
            int t = temp[ai];
            temp[ai] = temp[bi]; 
            temp[bi] += t;       
        } 
    }
    __syncthreads();

    temp[2*thid] = temp[2*thid + 1];
    
    if(thid == blockDim.x - 1) {
        temp[2*thid + 1] = blockSums[blockIdx.x];
    } else {
        temp[2*thid + 1] = temp[2*thid + 2];
    }

    g_odata[index] = temp[2*thid];  
    g_odata[index+1] = temp[2*thid+1];
}

__global__ void addOffsets(int * preScannedMask, int * blockScan) {

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(blockIdx.x == 0) return;

    preScannedMask[index] += blockScan[blockIdx.x-1];

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
    int input_size = (int)strlen(argv[1]);

    // GPU variables
    int * mask;
    int * host_mask;
    int * scannedMask;
    int * host_scannedMask;
    int * compactedMask;
    int * totalRuns;
    char * input;
    int * block_sums;
    int * scannedBlockSums;
    int * bs;
    int gridSize = (input_size + 1024 - 1) / 1024;

    //host_mask = (int *)malloc(input_size * sizeof(int));
    //host_scannedMask = (int *)malloc(input_size * sizeof(int));

    cudaMallocManaged(&input, input_size * sizeof(char));
    cudaMallocManaged(&mask, input_size * sizeof(int));
    cudaMallocManaged(&scannedMask, input_size * sizeof(int));
    cudaMallocManaged(&block_sums, gridSize * sizeof(int));
    cudaMallocManaged(&scannedBlockSums, gridSize * sizeof(int));
    cudaMallocManaged(&bs, gridSize * sizeof(int));

    cudaMemcpy(input, in, input_size * sizeof(char), cudaMemcpyHostToDevice);

    backWardMask<<<8, 1024>>>(in, mask, input_size);
    cudaDeviceSynchronize();

    prescan<<<gridSize, 1024>>>(scannedMask, mask, block_sums, input_size);
    cudaDeviceSynchronize();

    if(input_size > 2048) {
        
        // scan of block scans
        prescan<<<1,ceil(gridSize)>>>(scannedBlockSums, block_sums, bs, gridSize);
        cudaDeviceSynchronize();

        // add the offset
        addOffsets<<<gridSize, 2048>>>(scannedMask, scannedBlockSums);
    }

    //cudaMemcpy(host_mask, mask, input_size * sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(host_scannedMask, scannedMask, input_size * sizeof(int), cudaMemcpyDeviceToHost);

    // compactKernel<<<blocks, 512>>>();
    // cudaDeviceSynchronize();
    // scatterKernel<<<,>>>();
    // cudaDeviceSynchronize();

    for(int i = 0; i < input_size; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
    for(int i = 0; i < input_size; i++) {
        std::cout << mask[i] << " ";
    }
    // std::cout << std::endl;
    // for(int i = 0; i < input_size; i++) {
    //     std::cout << scannedMask[i] << " ";
    // }
    // std::cout << std::endl;
    
}