/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"
#include <limits.h>
#include <stdio.h>
#include <math.h>

int nextPowerOfTwo(int);

__global__ void shmemReduceMax(const float *d_in, float *d_out, const size_t n) {
  
  int myId = blockDim.x * blockIdx.x + threadIdx.x;
  int tid = threadIdx.x;

  extern __shared__ float sh_in[];
  if (myId < n)
    sh_in[tid] = d_in[myId];
  else
    sh_in[tid] = INT_MIN; // Initialize for last block -> edge case
    
  __syncthreads();

  for (unsigned int s = blockDim.x/2; s>0; s>>=1) {

    if (tid<s) {      
        float num1 = sh_in[tid];
        float num2 = sh_in[tid + s];      
        if (num2 > num1)
          sh_in[tid] = num2;
      //if (blockDim.x == 192 && s == 3)
        //      printf("tid=%d\t%f\t%f\t%f\n", tid, num1, num2, sh_in[tid]);
    }

    __syncthreads();  
  }

  if (tid == 0) {
    d_out[blockIdx.x] = sh_in[tid];
  }

}

__global__ void shmemReduceMin (const float *d_in, float *d_out, const size_t n) {
  
  int myId = blockDim.x * blockIdx.x + threadIdx.x;
  int tid = threadIdx.x;

  extern __shared__ float sh_in_min[];
  if (myId < n)
    sh_in_min[tid] = d_in[myId];
  else
    sh_in_min[tid] = INT_MAX; // For edge cases
  __syncthreads();

  for (unsigned int s = blockDim.x/2; s>0; s>>=1) {

    if (tid<s) {      
      float num1 = sh_in_min[tid];
      float num2 = sh_in_min[tid + s];
      
      if (num2 < num1)
          sh_in_min[tid] = num2;
    }

    __syncthreads();  

  }

  if (tid == 0) {
    d_out[blockIdx.x] = sh_in_min[tid];
  }

}

// Will modify this in future for optimization
// Create local histo arrays then reduce them to global ones
__global__ void simpleHist (const float *d_in, unsigned int *d_histo, float *d_min, const float D_RANGE, const int NUM_BINS, const int SIZE) {
  
  int myId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (myId < SIZE) {
      int bin = NUM_BINS * (d_in[myId] - *d_min) / D_RANGE;
      atomicAdd(&(d_histo[bin]), 1);
  }
}

__global__ void blolloch__exclusive_scan (unsigned int *d_in, int size, int steps) {
  
  int tid = threadIdx.x;
        
  extern __shared__ int sh_input[];
  sh_input[tid] = d_in[tid];
  __syncthreads();

  // reduction
  int gap = 1;
  for (size_t s = 0; s < steps; s++) {

    // find indices to operate on
    // Every thread with two different indices
    // dependent on step s

    int index2 = (tid+1) * gap * 2 - 1;
    int index1 = index2-gap;

    if (index2 < size) {
      if (index2 == size-1)
        sh_input[index2] = 0;
      else
        sh_input[index2] = sh_input[index2] + sh_input[index1];
    }
        
        gap<<=1;
    __syncthreads();
  }

  // downsweep
  for (size_t s = 0; s<steps; s++) {
        
        gap>>=1;
    int index2 = (tid+1) * gap * 2 - 1;
    int index1 = index2-gap;
        
    if (index2 < size) {
      int temp = sh_input[index2];
      sh_input[index2] += sh_input[index1];
      sh_input[index1] = temp;
    }
    
    __syncthreads();
  }
  
    //printf("%d\t%d\n", tid, sh_input[tid]);
  d_in[tid] = sh_input[tid];
}

__global__ void hillis_steele_inclusive_scan (unsigned int *d_in, int size, int steps) {
  
  int myId = threadIdx.x;

  extern __shared__ int sh_input[];
  // initialize sh_input
  sh_input[myId] = d_in[myId];
  __syncthreads();

  
  int sum;
  int gap = 1;
  for (int s=0; s<steps; s++) {

    if (myId - gap>=0) {
      sum = sh_input[myId] + sh_input[myId - gap];
    } else {
      sum = sh_input[myId];
    }

    gap <<= 1;    
    sh_input[myId] = sum;
    __syncthreads();
  }

    d_in[myId] = sh_input[myId];
}

void your_histogram_and_prefixsum(
  const float* const d_logLuminance, unsigned int* const d_cdf,
  float &min_logLum, float &max_logLum,
  const size_t numRows, const size_t numCols,
  const size_t numBins)
{

  const size_t numPixels = numRows * numCols;

  int threadsInBlock = 512;
  int numberOfBlocks = (numPixels + threadsInBlock - 1) / threadsInBlock;

  float *d_min_intermediate, *d_max_intermediate, *d_min, *d_max;
  int totalSize = threadsInBlock * sizeof(float);
    
  checkCudaErrors(cudaMalloc(&d_min_intermediate, numberOfBlocks * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_max_intermediate, numberOfBlocks * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_min, sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_max, sizeof(float)));
    
  shmemReduceMin<<<numberOfBlocks, threadsInBlock, totalSize>>>
            (d_logLuminance, d_min_intermediate, numPixels);

  shmemReduceMax<<<numberOfBlocks, threadsInBlock, totalSize>>>
            (d_logLuminance, d_max_intermediate, numPixels);
  
  
  int totalIntermediateElements = numberOfBlocks;
  
  // reduce the intermediate arrays to 1
  //threadsInBlock = totalIntermediateElements;
  
  threadsInBlock = 1;
  int q = nextPowerOfTwo(totalIntermediateElements);
  while (q != 0) {
      q--;
      threadsInBlock<<=1;
  }
  
  numberOfBlocks = 1;
  totalSize = threadsInBlock * sizeof(float);

  shmemReduceMin<<<numberOfBlocks, threadsInBlock, totalSize>>>
            (d_min_intermediate, d_min, totalIntermediateElements);

  shmemReduceMax<<<numberOfBlocks, threadsInBlock, totalSize>>>
            (d_max_intermediate, d_max, totalIntermediateElements);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));

  // free reduce variables
  // cudaFree(d_min); not yet
  cudaFree(d_max);
  cudaFree(d_max_intermediate);
  cudaFree(d_min_intermediate);
  const float RANGE = max_logLum - min_logLum;

  unsigned int *d_histo;
  
  checkCudaErrors(cudaMalloc(&d_histo, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));
    
  threadsInBlock = 512;
  numberOfBlocks = (numPixels + threadsInBlock - 1) / threadsInBlock;
  simpleHist<<<numberOfBlocks, threadsInBlock>>>
          (d_logLuminance, d_histo, d_min, RANGE, numBins, numPixels);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
    
    /*
    float *h_logLuminance = new float[numPixels];
    unsigned int *h_histo = new unsigned int[numBins];
    unsigned int *histo = new unsigned int[numBins];
    cudaMemcpy(h_logLuminance, d_logLuminance, sizeof(float)*numPixels, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_histo, d_histo, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < numBins; ++i) histo[i] = 0;
    for (size_t i = 0; i < numCols * numRows; ++i) {
        unsigned int bin = min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((h_logLuminance[i] - min_logLum) / RANGE * numBins));
        histo[bin]++;
    }
    
    for (int i = 0; i < numBins; i++){
        printf("%d\t%d\n", histo[i], h_histo[i]);
    }
    */

    //threadsInBlock = 9;
  threadsInBlock = numBins;
  numberOfBlocks = 1;
  
  float steps = log2((double)threadsInBlock);
  if (steps - (int)steps > 0)
      steps = (int)(steps + 1);

  blolloch__exclusive_scan<<<numberOfBlocks, threadsInBlock, numBins * sizeof(int)>>>
                      (d_histo, numBins, steps);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  
  checkCudaErrors(cudaMemcpy(d_cdf, d_histo, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToDevice));
    
  cudaFree(d_min);
  cudaFree(d_histo);
}

int nextPowerOfTwo(int n) {
  
  if (n == 1)
    return 1;

  int maxPossibleFactor = 0;
  while (1) {
    n = n/2;
    maxPossibleFactor++;
    if (n == 1)
      return maxPossibleFactor+1;
  }
  
}

