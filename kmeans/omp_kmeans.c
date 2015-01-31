/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         kmeans_clustering.c  (OpenMP version)                     */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "kmeans.h"
#include <pmmintrin.h>


#define SQR(a) ((a) * (a))

#define EUCLID_DIST_OPTIMIZED
//#define SSE_EUCLID_OPTIMIZED
//#define NO_OPTIMIZED
typedef union
{
	__m128 m;
	float f[4];
} ext;
/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__inline static
float euclid_dist_2(int    numdims,  /* no. dimensions */
                    float *coord1,   /* [numdims] */
                    float *coord2)   /* [numdims] */
{
    int i = 0;
    float ans=0.0;
#ifdef EUCLID_DIST_OPTIMIZED
	if(numdims > 16)
		for(i = 0;(i + 3) < numdims;i+=4)
			ans += SQR(coord1[i] - coord2[i]) + \
						SQR(coord1[i+1] - coord2[i+1]) + \
						SQR(coord1[i+2] - coord2[i+2]) + \
						SQR(coord1[i+3] - coord2[i+3]);
    for(; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);
#endif

#ifdef SSE_EUCLID_OPTIMIZED
	if(numdims >= 4)
	{
		ext x;
		__m128 tmp;
		__m128 *p1, *p2;
		for(i = 0;(i + 3) < numdims;i+=4)
		{
			p1 = (__m128*)&coord1[i];
			p2 = (__m128*)&coord2[i];
			tmp = _mm_sub_ps(*p1,*p2);
			/*
			tmp = _mm_set_ps(coord1[i] - coord2[i],\
															coord1[i+1] - coord2[i+1],\
															coord1[i+2] - coord2[i+2],\
															coord1[i+3] - coord2[i+3]);
			*/
			tmp = _mm_mul_ps(tmp,tmp);
			x.m = tmp;
			ans += (x.f[0] + x.f[1] + x.f[2]+ x.f[3]);
		}
	}
	for(;i<numdims;i++)
		ans += SQR(coord1[i]-coord2[i]);
#endif

#ifdef NO_OPTIMIZED
	for(;i<numdims;i++)
		ans += SQR(coord1[i] - coord2[i]);
#endif
    return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__inline static
int find_nearest_cluster(int     numClusters, /* no. clusters */
                         int     numCoords,   /* no. coordinates */
                         float  *object,      /* [numCoords] */
                         float **clusters)    /* [numClusters][numCoords] */
{
    int   index, i;
    float dist, min_dist;

    /* find the cluster id that has min distance to object */
    index    = 0;
    min_dist = euclid_dist_2(numCoords, object, clusters[0]);

    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, clusters[i]);
        /* no need square root */
        if (dist < min_dist) { /* find the min and its array index */
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}


/*----< kmeans_clustering() >------------------------------------------------*/
/* return an array of cluster centers of size [numClusters][numCoords]       */
float** omp_kmeans(int     is_perform_atomic, /* in: */
                   float **objects,           /* in: [numObjs][numCoords] */
                   int     numCoords,         /* no. coordinates */
                   int     numObjs,           /* no. objects */
                   int     numClusters,       /* no. clusters */
                   float   threshold,         /* % objects change membership */
                   int    *membership)        /* out: [numObjs] */
{

    int      i, j, k, index, loop=0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    float    delta;          /* % of objects change their clusters */
    unsigned *local_delta;
		int int_delta;
		unsigned int_threshold = threshold * numObjs; 
		
		float  **clusters;       /* out: [numClusters][numCoords] */
    float  **newClusters;    /* [numClusters][numCoords] */
    double   timing;

    int      nthreads;             /* no. threads */
    int    **local_newClusterSize; /* [nthreads][numClusters] */
    float ***local_newClusters;    /* [nthreads][numClusters][numCoords] */

//--------------------------------EDIT HERE-----------------------------//
#ifdef SSE_EUCLID_OPTIMIZED
		float **ori_objects = objects;
		objects = (float **) malloc(sizeof(float*) * numObjs * numCoords);
		for(i = 0; i < numObjs;i++)
		{
			posix_memalign((void **)&objects[i],16,sizeof(float) * numCoords);
			memcpy(objects[i],ori_objects[i],sizeof(float) * numCoords);
    }
#endif
//--------------------------------EDIT HERE----------------------------//
		nthreads = omp_get_max_threads();

		local_delta = (int*) malloc(sizeof(int) * nthreads);
		memset(local_delta,0,sizeof(int) * nthreads);
		is_perform_atomic = 0;
    /* allocate a 2D space for returning variable clusters[] (coordinates
       of cluster centers) */
    clusters    = (float**) malloc(numClusters * sizeof(float*));
    assert(clusters != NULL);
    
		//clusters[0] = (float*)  malloc(numClusters * numCoords * sizeof(float));
    //assert(clusters[0] != NULL);
    for (i=0; i<numClusters; i++)
				posix_memalign((void**) &clusters[i],16,sizeof(float) * numCoords);
        //clusters[i] = clusters[i-1] + numCoords;

    /* pick first numClusters elements of objects[] as initial cluster centers*/
    for (i=0; i<numClusters; i++)
        for (j=0; j<numCoords; j++)
            clusters[i][j] = objects[i][j];

    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    newClusters    = (float**) malloc(numClusters *            sizeof(float*));
    assert(newClusters != NULL);
    newClusters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
    assert(newClusters[0] != NULL);
    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + numCoords;

    if (!is_perform_atomic) {
        /* each thread calculates new centers using a private space,
           then thread 0 does an array reduction on them. This approach
           should be faster */
        local_newClusterSize    = (int**) malloc(nthreads * sizeof(int*));
        assert(local_newClusterSize != NULL);
        local_newClusterSize[0] = (int*)  calloc(nthreads*numClusters,
                                                 sizeof(int));
        assert(local_newClusterSize[0] != NULL);
        for (i=1; i<nthreads; i++)
            local_newClusterSize[i] = local_newClusterSize[i-1]+numClusters;

        /* local_newClusters is a 3D array */
        local_newClusters    =(float***)malloc(nthreads * sizeof(float**));
        assert(local_newClusters != NULL);
        local_newClusters[0] =(float**) malloc(nthreads * numClusters *
                                               sizeof(float*));
        assert(local_newClusters[0] != NULL);
        for (i=1; i<nthreads; i++)
            local_newClusters[i] = local_newClusters[i-1] + numClusters;
        for (i=0; i<nthreads; i++) {
            for (j=0; j<numClusters; j++) {
                local_newClusters[i][j] = (float*)calloc(numCoords,
                                                         sizeof(float));
                assert(local_newClusters[i][j] != NULL);
            }
        }
    }

    if (_debug) timing = omp_get_wtime();
    do {
        int_delta = 0; 

        if (is_perform_atomic) {
            #pragma omp parallel for \
                    private(i,j,index) \
                    firstprivate(numObjs,numClusters,numCoords) \
                    shared(objects,clusters,membership,newClusters,newClusterSize) \
                    schedule(static) \
                    reduction(+:int_delta)
            for (i=0; i<numObjs; i++) {
                /* find the array index of nestest cluster center */
                index = find_nearest_cluster(numClusters, numCoords, objects[i],
                                             clusters);

                /* if membership changes, increase delta by 1 */
                if (membership[i] != index) int_delta += 1;

                /* assign the membership to object i */
                membership[i] = index;

                /* update new cluster centers : sum of objects located within */
                #pragma omp atomic
                newClusterSize[index]++;
                for (j=0; j<numCoords; j++)
                    #pragma omp atomic
                    newClusters[index][j] += objects[i][j];
            }
        }
        else {
						int_delta = 0;
            #pragma omp parallel \
                    shared(objects,clusters,membership,local_newClusters,local_newClusterSize,local_delta)
            {
                int tid = omp_get_thread_num();
                #pragma omp for \
                            private(i,j,index) \
                            schedule(static) 
                for (i=0; i<numObjs; i++) {
                    /* find the array index of nestest cluster center */
                    index = find_nearest_cluster(numClusters, numCoords,
                                                 objects[i], clusters);

                    /* if membership changes, increase delta by 1 */
                    if (membership[i] != index) local_delta[tid] += 1;

                    /* assign the membership to object i */
                    membership[i] = index;

                    /* update new cluster centers : sum of all objects located
                       within (average will be performed later) */
                    local_newClusterSize[tid][index]++;
                    for (j=0; j<numCoords; j++)
                        local_newClusters[tid][index][j] += objects[i][j];
                }
            } /* end of #pragma omp parallel */

            /* let the main thread perform the array reduction */
            for (i=0; i<numClusters; i++) {
                for (j=0; j<nthreads; j++) {
                    newClusterSize[i] += local_newClusterSize[j][i];
                    local_newClusterSize[j][i] = 0;
                    for (k=0; k<numCoords; k++) {
                        newClusters[i][k] += local_newClusters[j][i][k];
                        local_newClusters[j][i][k] = 0.0;
                    }
                }
            }
        }

        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 1)
                    clusters[i][j] = newClusters[i][j] / newClusterSize[i];
                newClusters[i][j] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }
        for(i = 0; i < nthreads;i++)
				{
					int_delta += local_delta[i];
					local_delta[i] = 0;
				}
				//delta = (float)(int_delta) / numObjs;
    } while (int_delta > int_threshold && loop++ < 500);

    if (_debug) {
        timing = omp_get_wtime() - timing;
        printf("nloops = %2d (T = %7.4f)",loop,timing);
    }

    if (!is_perform_atomic) {
        free(local_newClusterSize[0]);
        free(local_newClusterSize);

        for (i=0; i<nthreads; i++)
            for (j=0; j<numClusters; j++)
                free(local_newClusters[i][j]);
        free(local_newClusters[0]);
        free(local_newClusters);
    }
//----------------EDIT HERE--------------//
#ifdef SSE_EUCLID_OPTIMIZED
		for(i = 0;i < numObjs;i++)
			free(objects[i]);
		free(objects);
		objects = ori_objects;
#endif
//----------------EDIT HERE--------------//		
		free(local_delta);
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}

