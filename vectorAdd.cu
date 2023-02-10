/*
 * vectorop = Vector scaling and translation: Y = a*X + B
 */

#include <stdio.h>
#include <cuda_runtime.h>

#include <chrono>

//#include <helper_cuda.h>
//#include <cuda_profiler_api.h>
/*
 * CUDA Kernel Device code
 */
__global__ void vectoropCUDA(const float *X, const float *B, float *Y, const float scalar,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements) {
    Y[i] = (X[i] * scalar) + B[i] + 0.0f;
  }
}

__global__ void vectoropCUDA_Streams(const float *X, const float *B, float *Y, const float scalar,
                          int numElements, int nStartIndex) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + nStartIndex;
  if ((i >= nStartIndex)&&(i < numElements)) {
    Y[i] = (X[i] * scalar) + B[i] + 0.0f;
    //printf("operation @ index %d, nStartIndex= %d\n",i,nStartIndex);
  }
}

void vectorop(const float *X, const float *B, float *Y, const float scalar,
                          int numElements) {
  for (int i = 0; i < numElements; i++) {
    Y[i] = (X[i] * scalar) + B[i] + 0.0f;
  }
}

/**
 * Host main routine
 */
int main(void) {
  cudaError_t err = cudaSuccess;
  
  // Print the vector length to be used, and compute its size
  int N = 29;
  int numElements = 1<<N; // 2^N elements
  size_t size = numElements * sizeof(float);
  printf("[Vector scaling and translation of %d elements (2^%d)]\n\n", numElements, N);

  // Scalar value (doesn't matter, just different from zero)
  float scalar = 3.37;

  // Allocate the host input vector X
  float *h_X = (float *)malloc(size);

  // Allocate the host input vector B
  float *h_B = (float *)malloc(size);

  // Allocate the host output vector Y
  float *h_Y = (float *)malloc(size);

  //Allocate host output vector Z used for measuring host execution time
  float *h_Z = (float *)malloc(size);

  // Verify that allocations succeeded
  if (h_X == NULL || h_B == NULL || h_Y == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_X[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // Allocate the device input vector X
  float *d_X = NULL;
  err = cudaMalloc((void **)&d_X, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector X (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector B
  float *d_B = NULL;
  err = cudaMalloc((void **)&d_B, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device output vector Y
  float *d_Y = NULL;
  err = cudaMalloc((void **)&d_Y, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector Y (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the host input vectors X and B in host memory to the device input
  // vectors in
  // device memory
  //printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector B from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  ///////////////////////////////////////////////////////////////////////////
  // main meat
  
  // HOST
  // Launch the Vector Add on Host Device and measure execution time
  
  int nHostCount = 1;
  
  printf("Performing host vector operation %d time(s)\n", nHostCount);
  auto hostmeasure_start = std::chrono::steady_clock::now();
  
  for (int j = 0; j < nHostCount; ++j) {
    vectorop(h_X, h_B, h_Z, scalar, numElements);
  }
    
  auto hostmeasure_end = std::chrono::steady_clock::now();
  double hostmeasure_msectotal = std::chrono::duration_cast<std::chrono::milliseconds>(hostmeasure_end-hostmeasure_start).count();
  double hostmeasure_msecpervector = hostmeasure_msectotal / nHostCount;
  
  printf("Execution time per operation: %.4f ms\n",hostmeasure_msecpervector);
  
  // CUDA DEFAULT STREAM
  // Launch the Vector Add CUDA Kernel
  // Measure time it took to execute the Kernel
  
  int nDeviceCount = 1000;
  
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
  printf("\nPerforming device vector operation %d times\n",nDeviceCount);
  auto measure_start = std::chrono::steady_clock::now();
  
  for (int h = 0; h < nDeviceCount; h++) {
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectoropCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_B, d_Y, scalar, numElements);
  }
  
  err = cudaDeviceSynchronize();
  auto measure_end = std::chrono::steady_clock::now();
  double measure_msectotal = std::chrono::duration_cast<std::chrono::milliseconds>(measure_end-measure_start).count();
  double measure_msecpervector = measure_msectotal / nDeviceCount;
  printf("Execution time per operation: %1.4f ms\n",measure_msecpervector);
  
  // CUDA NON-DEFAULT STREAMS
  
  int nStreamCount = 1000;
  int nStreams = 8;
  int nKernels = nStreams;
  int nNumElementsPerStream = numElements / nStreams;
  int blocksPerGridStream = (nNumElementsPerStream) / threadsPerBlock;
  
  //size_t streamSize = size / nStreams;
  printf("\nelements per stream: %d\n",nNumElementsPerStream);
  printf("blocks: %d\n", blocksPerGridStream);
  
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  cudaEvent_t *kernelEvent;
  kernelEvent = (cudaEvent_t *)malloc(nKernels*sizeof(cudaEvent_t));
  for (int i = 0;i<nKernels;i++) {
    cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming);
  }
  
  cudaStream_t *streams = (cudaStream_t *)malloc(nStreams * sizeof(cudaStream_t));
  for (int i = 0; i < nStreams; i++) {
    cudaStreamCreate(&(streams[i]));
  }
  printf("\nPerforming device NON-DEFAULT STREAMS (%d STREAMS) vector operation %d times\n",nStreams,nStreamCount);
  
  auto streammeasure_start = std::chrono::steady_clock::now();
  
  //cudaEventRecord(start_event,0);
  for (int k = 0; k < nStreamCount; k++) {
    for (int j = 0; j < nKernels; j++) {
      int idxOffset = j * nNumElementsPerStream;
      vectoropCUDA_Streams<<<blocksPerGridStream, threadsPerBlock, 0, streams[j]>>>(d_X, d_B, d_Y, scalar, (j+1)*nNumElementsPerStream, idxOffset);
      cudaEventRecord(kernelEvent[j],streams[j]);
      cudaStreamWaitEvent(streams[nStreams-1],kernelEvent[j],0);
    }
  }
  //cudaEventRecord(stop_event,0);
  //cudaEventSynchronize(stop_event);
  
  for (int i =0; i < nStreams; i++) {
    cudaStreamDestroy(streams[i]);
  }
  
  
  err = cudaDeviceSynchronize();
  auto streammeasure_end = std::chrono::steady_clock::now();
  double streammeasure_msectotal = std::chrono::duration_cast<std::chrono::milliseconds>(streammeasure_end-streammeasure_start).count();
  double streammeasure_msecpervector = streammeasure_msectotal / nStreamCount;;
  
  printf("Execution time per operation: %1.4f ms\n",streammeasure_msecpervector);
  
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorop kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < nKernels; i++) {
    cudaEventDestroy(kernelEvent[i]);
  }
  free(kernelEvent);
  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);



  ///////////////////////////////////////////////////////////////////////////

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  //printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_Y, d_Y, size, cudaMemcpyDeviceToHost);

  free(streams);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_X[i]*scalar + h_B[i] - h_Y[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
    //printf("CUDA: %f, HOST: %f\n", h_Y[i], h_Z[i]);
  }
  printf("Test PASSED\n");
  

  // Free device global memory
  err = cudaFree(d_X);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_B);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_Y);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free host memory
  free(h_X);
  free(h_B);
  free(h_Y);
  free(h_Z);

  printf("Done\n");
  return 0;
}
