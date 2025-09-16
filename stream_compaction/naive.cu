#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
    //    __global__ void kernPadToPo2(int n_pad, int n, int* odata, const int* idata) {
    //        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    //        if (index >= n_pad) {
    //          return;
    //        }
    //        if (index < n) {
    //            odata[index] = idata[index];
    //        } else {
    //            odata[index] = 0;
    //        }
				//}

        __global__ void kernNaiveScan(int n, int offset, int *odata, const int *idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            if (index >= offset) {
                odata[index] = idata[index - offset] + idata[index];
            } else {
                odata[index] = idata[index];
            }
				}

        __global__ void kernIncToExc(int n, int* odata, const int* idata) {
          int index = blockIdx.x * blockDim.x + threadIdx.x;
          if (index >= n) return;

          if (index == 0) {
            odata[index] = 0;
          }
          else {
            odata[index] = idata[index - 1];
          }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            // TODO
						int* dev_idata1;
            int* dev_idata2;

						//int n_pad = 1;
						//while (n_pad < n) n_pad <<= 1;
      //      int* h_data = new int[n_pad];
      //      for (int i = 0; i < n; i++) {
      //        h_data[i] = idata[i];
      //      }
      //      for (int i = n; i < n_pad; i++) {
      //        h_data[i] = 0;
      //      }

            cudaMalloc((void**)&dev_idata1, n * sizeof(int));
            cudaMalloc((void**)&dev_idata2, n * sizeof(int));

            cudaMemcpy(dev_idata1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);


            int offset = 1;

            timer().startGpuTimer();
            for (int offset = 1; offset < n; offset <<= 1) {
                kernNaiveScan<<<fullBlocksPerGrid, blockSize>>>(n, offset, dev_idata2, dev_idata1);
                //cudaDeviceSynchronize();
                checkCUDAError("kernel scan error!");

								std::swap(dev_idata1, dev_idata2);
            }


						kernIncToExc << <fullBlocksPerGrid, blockSize >> > (n, dev_idata2, dev_idata1);
            timer().endGpuTimer();
						//cudaDeviceSynchronize();
						checkCUDAError("kernel inc to exc error!");

            cudaMemcpy(odata, dev_idata2, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata1);
						cudaFree(dev_idata2);
						//delete[] h_data;

        }
    }
}
