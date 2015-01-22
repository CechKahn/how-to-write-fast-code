#include <stdio.h>
#include <omp.h>
#include <sys/time.h>

#define NUM_STEPS 80000000
#define STEP (double)1/NUM_STEPS
#define NUM_OF_THREADS 8

double serial_sum()
{
	unsigned long i = 0;
	double x = 0;
	double pi = 0;
	double sum = 0;
	for(i = 0;i < NUM_STEPS;i++)
	{
		x = (i + 0.5) * STEP;
		sum = sum + 4.0 / (1.0 + x * x);
	}
	pi = STEP * sum;
	return pi;
}

double omp_sum()
{
	double sum[NUM_OF_THREADS];
	double pi = 0;
	unsigned long part_of_work = NUM_STEPS / NUM_OF_THREADS;
	omp_set_num_threads(NUM_OF_THREADS);
#pragma omp parallel
{
	int id = omp_get_thread_num();
	unsigned long i = 0;
	unsigned long indi_work_start = id * part_of_work;
	unsigned long indi_work_end = (id + 1) * part_of_work;
	printf("Thread %d calculating from (%lu, %lu)\n",id,indi_work_start,indi_work_end);
	sum[id] = 0;
	double x;
	for(i = indi_work_start; i < indi_work_end;i++)
	{
		x = (i + 0.5) * STEP;
		sum[id] = sum[id] + ((double)4.0/(1.0 + x * x));	
	}
	sum[id]*=STEP;
	printf("Thread %d get result %lf\n",id,sum[id]);
}

	for(int count = 0;count < NUM_OF_THREADS;count++)
		pi+=sum[count];
	return pi;

}
int main()
{
	struct timeval stop,start;
	gettimeofday(&start,NULL);
	printf("pi is esitmated to be %lf\n",serial_sum());
	gettimeofday(&stop,NULL);
	printf("Serially It took %lu ms\n", ((stop.tv_sec - start.tv_sec) * 1000000L + stop.tv_usec) - start.tv_usec);

	gettimeofday(&start,NULL);
	printf("pi is estimated to be %lf\n",omp_sum());
	gettimeofday(&stop,NULL);
	printf("Using Openmp it took %lu ms\n",((stop.tv_sec - start.tv_sec) * 1000000L + stop.tv_usec) - start.tv_usec);
	return 0;
}

