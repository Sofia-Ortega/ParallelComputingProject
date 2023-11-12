#include <iostream>
#include <cstdlib>

__global__ void oddevenSort(int* array, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int phase, tmp;

    for (phase = 0; phase < size; phase++) {
        if (phase % 2 == 0) { // Even phase
            if (tid % 2 == 0 && tid < size - 1) {
                if (array[tid] > array[tid + 1]) {
                    // Swap elements
                    tmp = array[tid];
                    array[tid] = array[tid + 1];
                    array[tid + 1] = tmp;
                }
            }
        } else { // Odd phase
            if (tid % 2 == 1 && tid < size - 1) {
                if (array[tid] > array[tid + 1]) {
                    // Swap elements
                    tmp = array[tid];
                    array[tid] = array[tid + 1];
                    array[tid + 1] = tmp;
                }
            }
        }
        __syncthreads(); // Synchronize threads within the block
    }
}

int main() {
    int arraySize = 1024;
    int* h_array = new int[arraySize];
    int* d_array;

    for (int i = 0; i < arraySize; i++) {
        h_array[i] = rand() % 1000;
    }

    cudaMalloc(&d_array, sizeof(int) * arraySize);
    cudaMemcpy(d_array, h_array, sizeof(int) * arraySize, cudaMemcpyHostToDevice);

    int numBlocks = 4; // Choose the number of blocks as needed
    int threadsPerBlock = arraySize / numBlocks;

    oddevenSort<<<numBlocks, threadsPerBlock>>>(d_array, arraySize);
    cudaDeviceSynchronize();

    cudaMemcpy(h_array, d_array, sizeof(int) * arraySize, cudaMemcpyDeviceToHost);

    // Print sorted array
    printf("Sorted:\n");
    for (int i = 0; i < arraySize; i++) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    bool isSorted = true;
    for(int i = 1; i < arraySize; i++) {
        if(h_array[i - 1] > h_array[i]) {
            printf("[FAIL] is not sorted [%i]\n", i);
            isSorted = false;
            break;
        }
    }

    if(isSorted)
        printf("Success!\n");



    delete[] h_array;
    cudaFree(d_array);

    return 0;
}
