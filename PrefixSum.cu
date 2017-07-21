
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include<random>
using namespace std;

__global__ void prescan(float *g_odata, float *g_idata, int n)
{
	extern __shared__ float temp[];  // allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;
	
	temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
	temp[2 * thid + 1] = g_idata[2 * thid + 1];
	//printf("%d - %f - %f \n", thid, g_odata[2 * thid], g_odata[2 * thid + 1]);
	//printf("%d - %f - %f \n", thid, g_idata[2 * thid], g_idata[2 * thid + 1]);
	for (int d = n >> 1; d > 0; d >>= 1)                    // build sum in place up the tree
	{
		__syncthreads();
		if (thid < d)
		{
			
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;


			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	

	if (thid == 0) { temp[n - 1] = 0; } // clear the last element


	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			

				int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;


			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	
	g_odata[2 * thid] = temp[2 * thid]; // write results to device memory
	g_odata[2 * thid + 1] = temp[2 * thid + 1];
	
//	printf("%d - %f - %f \n", thid, g_odata[2 * thid], g_odata[2 * thid + 1]);
	//printf("%d - %f - %f \n", thid, g_idata[2 * thid], g_idata[2 * thid + 1]);
}

float * getData(int size){
	
	float *data = new float [size];
	for (int i = 0; i < size; i++){
		data[i] = std::rand()%10;
	}
	return data;
}

void displayData(float * data,int size){
	for (int i= 0; i < size; i++){
		printf("%f\n", data[i]);
	}
}
int main(){

	int size = 128;
	float *input;
	float *output = new float[size];
	float *inp_dev, *out_dev;

	input = getData(size);
//	displayData(input, size);

	cudaMalloc(&inp_dev, size*sizeof(float));
	
	cudaMalloc(&out_dev, size*sizeof(float));

	cudaError_t cudaStatus =  cudaMemcpy(inp_dev, input, size*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaMemset(out_dev, 0, size*sizeof(float));
	prescan << <1, size/2, size*sizeof(float) >> >(out_dev,inp_dev, size);
	
	cudaDeviceSynchronize();
	cudaMemcpy(output, out_dev, size*sizeof(float), cudaMemcpyDeviceToHost);

	cout << "\nFinal Output\n";
	displayData(output, size);
	
	Error:

	getchar();

	return cudaStatus;
}