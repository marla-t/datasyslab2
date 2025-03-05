#include <stdlib.h>
#include <math.h>
#include <iostream>

#define ROW_TILE_N  32
#define COL_TILE_N  32
#define EPSILON         (1e-6)

__global__ void matrix_multiply(float *A, float *B, float* C, int N)
{
  __shared__ float locA[ROW_TILE_N][COL_TILE_N], locB[ROW_TILE_N][COL_TILE_N];
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float tmp = 0;

  for(int i = 0; i < N/COL_TILE_N; i++) {
    locA[threadIdx.y][threadIdx.x] = A[row * N + i * ROW_TILE_N + threadIdx.x];
    locB[threadIdx.y][threadIdx.x] = B[(i * COL_TILE_N + threadIdx.y) * N + col];
    __syncthreads();
    for(int j = 0; j < COL_TILE_N;  j++)
       tmp += locA[threadIdx.y][j] * locB[j][threadIdx.x];
    __syncthreads();
  }  
  C[row * N + col] = tmp;
}

void init_mat(float* M, int N) {
  for(int i = 0; i < N*N; i++)
      M[i] = (float(rand()%255)/255.0);
}

void matrix_multiply_cpu(float *A, float *B, float* C, int N)
{
  for(int i = 0; i < N; i++)
    for(int j = 0; j < N; j++){
      float tmp = 0.0;
      for(int k = 0; k < N; k++)
        tmp += A[i * N + k] * B[k * N + j];     
      C[i * N + j] = tmp;
    }
}

float maxDiff(float* A1, float* A2, int N){
  float maxDiff = A1[0] - A2[0];
  for(int i = 0; i < N*N; i++) {
      float diff = abs(A1[i] - A2[i]);
      if( diff > maxDiff)
          maxDiff = diff;
  }  
  return maxDiff;
}


int main(int argc, char *argv[])
{
  if (argv[1] == NULL) { fprintf(stderr,"Please provide matrix dimension in the second argument\n ./gemm.cu N\n"); exit(0); }
  int N = atoi(argv[1]);
  int A_size = N * N, B_size = N * N, C_size = N * N;
  float *A, *B, *C, *C_cpu, *A_gpu, *B_gpu, *C_gpu;

  /* Allocate memory on device */
  cudaMalloc(&A_gpu, A_size*sizeof(float));
  cudaMalloc(&B_gpu, B_size*sizeof(float));
  cudaMalloc(&C_gpu, C_size*sizeof(float));

  /* Allocate memory on host */
  A = (float*) malloc(A_size*sizeof(float));
  B = (float*) malloc(A_size*sizeof(float));
  C = (float*) malloc(A_size*sizeof(float));
  C_cpu = (float*) malloc(A_size*sizeof(float));

  /* Initialize matrix A, B */
  init_mat(A, N); init_mat(B, N);

  /* Copy A and B to device */
  cudaMemcpy(A_gpu, A, A_size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, B_size*sizeof(float), cudaMemcpyHostToDevice);


  /* Launch the matmul kernel */
  dim3 dim_grid(N/COL_TILE_N, N/ROW_TILE_N, 1);
  dim3 dim_block(COL_TILE_N, ROW_TILE_N, 1);
  matrix_multiply<<<dim_grid, dim_block>>>(A_gpu, B_gpu, C_gpu, N);

  /* Synchronize */
  cudaDeviceSynchronize();
  
  /* Copy result matrix C back */
  cudaMemcpy(C_cpu, C_gpu, C_size*sizeof(float), cudaMemcpyDeviceToHost);

  /* Compute CPU version */
  matrix_multiply_cpu(A, B, C, N);

  /* Check to make sure both are correct (within EPSILOON) */
  if(fabsf(maxDiff(C_cpu, C, N)) <= (float)EPSILON )
     std::cout << "All correct" << std::endl;
  else
     std::cout << "Failed" << std::endl;


  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C_cpu);
  cudaFree(C_gpu);
  
  return 0; 
}