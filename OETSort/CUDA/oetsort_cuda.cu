#include <stdio.h>
#include <stdlib.h>

__global__ void oddEvenSort(int* arr, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) { // Even phase
            if (tid % 2 == 0 && tid < n - 1) {
                if (arr[tid] > arr[tid + 1]) {
                    int temp = arr[tid];
                    arr[tid] = arr[tid + 1];
                    arr[tid + 1] = temp;
                }
            }
        } else { // Odd phase
            if (tid % 2 != 0 && tid < n - 1) {
                if (arr[tid] > arr[tid + 1]) {
                    int temp = arr[tid];
                    arr[tid] = arr[tid + 1];
                    arr[tid + 1] = temp;
                }
            }
        }
        __syncthreads();
    }
}

int main(int argc, char** argv) {
    // argv:
    // 0            1           2                3
    // oetsort_cuda num_threads num_vals_to_sort [optional: printArray]

    if(argc != 3 && argc != 4) {
        printf("Incorrect argumant usage\n");
        printf("oetsort_cuda num_threads num_vals_to_sort [optional: print_array]\n");
        return -1;
    }

    int num_threads = atoi(argv[1]);
    int n = atoi(argv[2]); // Size of the array
    bool printArray = false;

    if(argc == 4) {
        printArray = atoi(argv[3]);
    }

    printf("Sorting %i values with %i threads\n", n, num_threads);


    int* h_array = (int*)malloc(n * sizeof(int));
    int* d_array;

    // intialize local array
    for (int i = 0; i < n; i++) {
        h_array[i] = rand() % 100;
    }

    if(printArray) {
        printf("unsorted array: \n");
        for (int i = 0; i < n; i++) {
            printf("%i: ", i);
            printf("%d\n", h_array[i]);
        }
        printf("\n");
    }

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_array, n * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    int numBlocks = n  / num_threads;
    if(n % num_threads != 0 ) 
        numBlocks++;


    oddEvenSort<<<numBlocks, num_threads>>>(d_array, n);

    // Copy the sorted data back to the host
    cudaMemcpy(h_array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);

    // check if array is sorted
    bool isSorted = true;
    for(int i = 1; i < n; i++) {
        if(h_array[i - 1] > h_array[i]) {
            printf("[ERROR] [%i, %i]: %i, %i Incorrect values\n", i - 1, i, h_array[i - 1], h_array[i]);
            isSorted = false;
            break;
        }
    }

    if(isSorted) {
        printf("Great Success! It is Sorted :D \n");
    } else {
        printf("Something went terribly wrong...\n");
    }

    // Print the sorted array
    if(printArray) {
        printf("Supposeldy Sorted Array: o.o \n");
        for (int i = 0; i < n; i++) {
            printf("%i: ", i);
            printf("%d\n", h_array[i]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_array);
    free(h_array);

    return 0;
}
