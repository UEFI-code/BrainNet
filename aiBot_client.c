#include <unistd.h>  
#include <stdlib.h>  
#include <stdio.h>
#include <sys/shm.h>
#define N 256
float* iobuf = 0;
float* weights = 0;
float* cellConsume = 0;
int main()
{
	int shmid = shmget((key_t)2333, sizeof(float)*N, 0777|IPC_CREAT);
        iobuf = (float *)shmat(shmid, 0, 0);
        shmid = shmget((key_t)2334, sizeof(float)*N*(N-1), 0777|IPC_CREAT);
        weights = (float *)shmat(shmid, 0, 0);
        shmid = shmget((key_t)2335, sizeof(float)*N, 0777|IPC_CREAT);
        cellConsume = (float *)shmat(shmid, 0, 0);
	while(1)
	{

		for(int i=0;i<N;i++)
			printf("%f ",iobuf[i]);
		printf("\n\n\n");
		sleep(1);
		iobuf[0] = (rand()%1000)/100.0;
	}
}
