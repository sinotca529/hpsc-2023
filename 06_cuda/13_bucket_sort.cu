#include <cstdio>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <assert.h>

typedef struct {
  int *array;
  int len;
} Array;

Array new_managed_array(int len) {
  int *array;
  cudaMallocManaged(&array, len * sizeof(int));
  Array r = { array, len };
  return r;
}

__global__ void bucket_sort(Array keys, Array bucket) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int b[];

  // zero clear bucket
  if (i < bucket.len) bucket.array[i] = 0;
  __syncthreads();

  // fill bucket
  if (i < keys.len) atomicAdd(&(bucket.array[keys.array[i]]), 1);
  __syncthreads();

  // fill sorted keys
  if (i < keys.len) {
    int acc = 0;
    for (int bid = 0; bid < bucket.len; ++bid) {
      acc += bucket.array[bid];
      if (i < acc) {
        keys.array[i] = bid;
        break;
      }
    }
  }
}

int main() {
  const int N = 200;
  const int M = 1024;

  int n = 50;
  int range = 5;

  Array keys = new_managed_array(n);
  for (int i=0; i<n; i++) {
    keys.array[i] = rand() % range;
    printf("%d ",keys.array[i]);
  }
  printf("\n");

  Array bucket = new_managed_array(range);

  assert(std::max(keys.len, bucket.len) < N * M);
  bucket_sort<<<(N+M-1)/M, M, range*sizeof(int)>>>(keys, bucket);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",keys.array[i]);
  }
  printf("\n");
}
