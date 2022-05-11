#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<unistd.h>
#define N 64
#define CellEnergy 100
#define MaxOut 10.0
#define RecoverySpeed 5.0
#define tiredRecoveryRate 0.6
float iobuf[N] = {0.0};
float iobuf_new[N] = {0.0};
float weights[N * (N-1)] = {0.0};
float cellConsume[N] = {0.0};
float Activation(float x, int i)
{
	if(x>MaxOut) x = (x/(x+1))*MaxOut;
	if(x<0-MaxOut) x = (x/(1-x))*MaxOut;
	if(CellEnergy - cellConsume[i] > 0)
	{
		x = x * (CellEnergy - cellConsume[i]) / CellEnergy;
		cellConsume[i] += x - RecoverySpeed;
		if(cellConsume[i] < 0) cellConsume[i] = 0;
		return x;
	}
	else
	{
		cellConsume[i] = cellConsume[i] * tiredRecoveryRate;
		return 0.0;
	}
}
int main()
{
	srand((unsigned)time(NULL));
	for(int i=0;i<(N * (N-1));i++)
	{
		weights[i] = (rand()%200)/100.0;
		if((rand()%2) == 0)
			weights[i] = 0 - weights[i];
	}
	iobuf[0] = (rand()%200)/100.0;
	while(1)
	{
		for(int i=0;i<N;i++)
		{
			iobuf_new[i] = 0.0;
			for(int j=0;j<N-1;j++)
			{
				if(j<i)
					iobuf_new[i] += iobuf[j] * weights[i*(N-1)+j];
				else
					iobuf_new[i] += iobuf[j+1] * weights[i*(N-1)+j];
			}
			iobuf_new[i] = Activation(iobuf_new[i],i);
			printf("%.2f ",iobuf_new[i]);
		}
		for(int i=0;i<N;i++)
			iobuf[i] = iobuf_new[i];
		printf("\n");
		usleep(100);
	}
	
}
