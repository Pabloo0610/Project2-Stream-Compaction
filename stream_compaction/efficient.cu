#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        //__global__ void kernPadToPo2(int n_pad, int n, int* odata, const int* idata) {
        //  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        //  if (index >= n_pad) {
        //    return;
        //  }
        //  if (index < n) {
        //    odata[index] = idata[index];
        //  }
        //  else {
        //    odata[index] = 0;
        //  }
        //}

        __global__ void kernUpSweepScan(int n, int offset, int* data) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) {
            return;
          }
					//int mod_index = index * 2 * offset;
     //     if (mod_index + 2 * offset - 1 < n) {
     //       data[mod_index+ 2*offset - 1] += data[mod_index + offset - 1];
     //     }
          int bi = (index + 1) * 2 * offset - 1;
					int ai = bi - offset;
					data[bi] += data[ai];
        }

        __global__ void kernSetZero(int n, int* data) {
          if (threadIdx.x == 0 && blockIdx.x == 0) {
            data[n - 1] = 0;
          }
        }

        __global__ void kernDownSweepScan(int n, int offset, int* data) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) {
            return;
          }
					//int mod_index = index * 2 * offset;
     //     if (mod_index + 2 * offset - 1 < n) {
     //       int t = data[mod_index + offset - 1];
     //       data[mod_index + offset - 1] = data[mod_index + 2 * offset - 1];
     //       data[mod_index + 2 * offset - 1] += t;
					//}
					int bi = (index + 1) * 2 * offset - 1;
					int ai = bi - offset;
					int t = data[ai];
					data[ai] = data[bi];
					data[bi] += t;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

          int n_pad = 1;
          while (n_pad < n) n_pad <<= 1;

					int* h_data = new int[n_pad];
          for (int i = 0; i < n; i++) {
            h_data[i] = idata[i];
          }
          for (int i = n; i < n_pad; i++) {
            h_data[i] = 0;
          }

					int* dev_data;

					cudaMalloc((void**)&dev_data, n_pad * sizeof(int));
          cudaMemcpy(dev_data, h_data, n_pad * sizeof(int), cudaMemcpyHostToDevice);

          int blockSize = 256;
          dim3 fullBlocksPerGrid((n_pad + blockSize - 1) / blockSize);

          timer().startGpuTimer();
					for (int offset = 1; offset < n_pad; offset <<= 1) {
            int elems = n_pad / (2 * offset);
            //int blocks = (elems + blockSize - 1) / blockSize;
						dim3 blocks((elems + blockSize - 1) / blockSize);
						kernUpSweepScan << <blocks, blockSize >> > (elems, offset, dev_data);
          }
          kernSetZero << <1, 1 >> > (n_pad, dev_data);
          for (int offset = n_pad >> 1; offset > 0; offset >>= 1) {
            int elems = n_pad / (2 * offset);
            //int blocks = (elems + blockSize - 1) / blockSize;
						dim3 blocks((elems + blockSize - 1) / blockSize);
            kernDownSweepScan << <blocks, blockSize >> > (elems, offset, dev_data);
					}
          timer().endGpuTimer();

					cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

          cudaFree(dev_data);
					delete[] h_data;
				}
        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {

          int n_pad = 1;
          while (n_pad < n) n_pad <<= 1;

          int* h_data = new int[n_pad];
          for (int i = 0; i < n; i++) {
            h_data[i] = idata[i];
          }
          for (int i = n; i < n_pad; i++) {
            h_data[i] = 0;
          }

          int* dev_data;
					int* dev_bool;
          int* dev_scan;
					int* dev_comp;

					cudaMalloc((void**)&dev_data, n_pad * sizeof(int));
					cudaMalloc((void**)&dev_bool, n_pad * sizeof(int));
					cudaMalloc((void**)&dev_scan, n_pad * sizeof(int));
					cudaMalloc((void**)&dev_comp, n_pad * sizeof(int));

					cudaMemcpy(dev_data, h_data, n_pad * sizeof(int), cudaMemcpyHostToDevice);
					int blockSize = 128;
					dim3 fullBlocksPerGrid((n_pad + blockSize - 1) / blockSize);

					Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n_pad, dev_bool, dev_data);

          timer().startGpuTimer();
					cudaMemcpy(dev_scan, dev_bool, n_pad * sizeof(int), cudaMemcpyDeviceToDevice);
          for (int offset = 1; offset < n_pad; offset <<= 1) {
            int elems = n_pad / (2 * offset);
            //int blocks = (elems + blockSize - 1) / blockSize;
            dim3 blocks((elems + blockSize - 1) / blockSize);
            kernUpSweepScan << <blocks, blockSize >> > (elems, offset, dev_scan);
          }
          kernSetZero << <1, 1 >> > (n_pad, dev_scan);
          for (int offset = n_pad >> 1; offset > 0; offset >>= 1) {
            int elems = n_pad / (2 * offset);
            //int blocks = (elems + blockSize - 1) / blockSize;
            dim3 blocks((elems + blockSize - 1) / blockSize);
            kernDownSweepScan << <blocks, blockSize >> > (elems, offset, dev_scan);
          }

					Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n_pad, dev_comp, dev_data, dev_bool, dev_scan);

          timer().endGpuTimer();

					int cnt = 0;
          cudaMemcpy(&cnt, dev_scan + n_pad - 1, sizeof(int), cudaMemcpyDeviceToHost);

					cudaMemcpy(odata, dev_comp, cnt * sizeof(int), cudaMemcpyDeviceToHost);

          cudaFree(dev_data);
          cudaFree(dev_bool);
          cudaFree(dev_scan);
          cudaFree(dev_comp);
					delete[] h_data;

          return cnt;
        }
    }
}
