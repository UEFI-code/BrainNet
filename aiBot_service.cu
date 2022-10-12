#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<unistd.h>
#include <sys/shm.h>
#define N 256
#define PoolEnergy 1.0
#define CellEnergy 25.0
#define MaxOut 10.0
#define RecoverySpeed 3.0
#define tiredRecoveryRate 0.6
float *iobuf = 0;
float *iobuf_gpu = 0;
float *weights = 0;
float *weights_gpu = 0;
float *cellConsume = 0;
float *cellConsume_gpu = 0;
__global__ void Neural(float *iobuf_gpu, float *weights_gpu, float *cellConsume_gpu, float exciteNum) 
{
	float x = 0.0;
	bool flag = true;
	for(int i=0;i<N-1;i++)
        {
		if(i<threadIdx.x)
			x += iobuf_gpu[i] * weights_gpu[threadIdx.x*(N-1)+i];
		else
			x += iobuf_gpu[i+1] * weights_gpu[threadIdx.x*(N-1)+i];
        }
	if(threadIdx.x < exciteNum)
	{
		flag = true;
		if(x>0.0)
		{
			if(x>MaxOut) x = (x/(x+1))*MaxOut;
		}
		else x = 0.0;
	}
	else
	{
		flag = false;
		x = 0 - x;
		if(x<0.0)
		{
			if(x<0-MaxOut) x = (x/(1-x))*MaxOut;
		}
		else x = 0.0;
	}
	if(CellEnergy - cellConsume_gpu[threadIdx.x] > 0)
	{
		x = x * (CellEnergy - cellConsume_gpu[threadIdx.x]) / CellEnergy;
		if(flag)
			cellConsume_gpu[threadIdx.x] -= (RecoverySpeed - x) * PoolEnergy;
		else
			cellConsume_gpu[threadIdx.x] -= (RecoverySpeed + x) * PoolEnergy;
		if(cellConsume_gpu[threadIdx.x] < 0) cellConsume_gpu[threadIdx.x] = 0;
	}
	else
	{
		cellConsume_gpu[threadIdx.x] = cellConsume_gpu[threadIdx.x] * tiredRecoveryRate;
		x = 0.0;
	}
	iobuf_gpu[threadIdx.x] = x;
}

int main()
{
	srand((unsigned)time(NULL));
	int shmid = shmget((key_t)2333, sizeof(float)*N, 0777|IPC_CREAT);
        iobuf = (float *)shmat(shmid, 0, 0);
	shmid = shmget((key_t)2334, sizeof(float)*N*(N-1), 0777|IPC_CREAT);
	weights = (float *)shmat(shmid, 0, 0);
	shmid = shmget((key_t)2335, sizeof(float)*N, 0777|IPC_CREAT);
        cellConsume = (float *)shmat(shmid, 0, 0);
	for(int i=0;i<N;i++)
	{
		iobuf[i] = 0;
		cellConsume[i] = 0;
	}
	for(int i=0;i<(N * (N-1));i++)
		weights[i] = (rand()%100)/100.0;
	cudaMalloc(&iobuf_gpu, sizeof(float)*N);
	cudaMalloc(&weights_gpu, sizeof(float)*N*(N-1));
	cudaMalloc(&cellConsume_gpu, sizeof(float)*N);
	while(1)
	{
		cudaMemcpy(weights_gpu, weights, sizeof(float)*N*(N-1), cudaMemcpyHostToDevice);
        	cudaMemcpy(cellConsume_gpu, cellConsume, sizeof(float)*N, cudaMemcpyHostToDevice);
                cudaMemcpy(iobuf_gpu, iobuf, sizeof(float)*N, cudaMemcpyHostToDevice);
		Neural<<<1,N>>>(iobuf_gpu,weights_gpu,cellConsume_gpu,N-4);
		cudaMemcpy(iobuf, iobuf_gpu, sizeof(float)*N, cudaMemcpyDeviceToHost);
        	cudaMemcpy(cellConsume, cellConsume_gpu, sizeof(float)*N, cudaMemcpyDeviceToHost);
        	
		cudaMemcpy(weights, weights_gpu, sizeof(float)*N, cudaMemcpyDeviceToHost);
	}
		
}
