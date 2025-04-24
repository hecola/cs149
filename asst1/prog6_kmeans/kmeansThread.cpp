#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <immintrin.h>

#include "CycleTimer.h"

using namespace std;
#define MAX_THREADS 16
typedef struct {
  // Control work assignments
  int start, end;
  int M, N, K;
  // Shared by all functions
  double *data;
  double *clusterCentroids;
  int *clusterAssignments;
  double *currCost;
  // for threads
  int step, startK, endK, startM, endM, threadID;
  void* temp; // for computeAssignments_bythread
  int *thread_counts; //for computeCentroids_bythread
  double *thread_centroids; // for computeCentroids
  double* thread_accum; // for computeCost
} WorkerArgs;


/**
 * Checks if the algorithm has converged.
 * 
 * @param prevCost Pointer to the K dimensional array containing cluster costs 
 *    from the previous iteration.
 * @param currCost Pointer to the K dimensional array containing cluster costs 
 *    from the current iteration.
 * @param epsilon Predefined hyperparameter which is used to determine when
 *    the algorithm has converged.
 * @param K The number of clusters.
 * 
 * NOTE: DO NOT MODIFY THIS FUNCTION!!!
 */
static bool stoppingConditionMet(double *prevCost, double *currCost,
                                 double epsilon, int K) {
  for (int k = 0; k < K; k++) {
    //printf("k: %d, prevCost: %f, currCost: %f\n", k, prevCost[k], currCost[k]);
    if (abs(prevCost[k] - currCost[k]) > epsilon)
      return false;
  }
  return true;
}

/**
 * Computes L2 distance between two points of dimension nDim.
 * 
 * @param x Pointer to the beginning of the array representing the first
 *     data point.
 * @param y Poitner to the beginning of the array representing the second
 *     data point.
 * @param nDim The dimensionality (number of elements) in each data point
 *     (must be the same for x and y).
 */
double dist(double *x, double *y, int nDim) {
  double accum = 0.0;
  __m256d va = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);
  __m256d vb = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);
  __m256d vc = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);
  __m256d result = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);
  for (int i = 0; i < nDim; i+=4) {
    va = _mm256_loadu_pd(&x[i]);
    vb = _mm256_loadu_pd(&y[i]);
    result = _mm256_sub_pd(va, vb);
    vc = _mm256_fmadd_pd(result, result, vc); // FMA
  }
  __m128d low = _mm256_extractf128_pd(vc, 0);
  __m128d high = _mm256_extractf128_pd(vc, 1);
  __m128d sum128 = _mm_add_pd(low, high);
  sum128 = _mm_hadd_pd(sum128, sum128);
  return sqrt(_mm_cvtsd_f64(sum128));
}

void computeAssignments_bythread(WorkerArgs *const args) {
  int M = args->M;
  int N = args->N;
  int K = args->K;
  int startM = args->startM;
  int endM = args->endM;
  int startK = args->startK;
  int endK = args->endK;
  int step = args->step;  
  double *minDist = (double*)args->temp;
  double *data = args->data;
  double *clusterCentroids = args->clusterCentroids;
  int *clusterAssignments = args->clusterAssignments;
  // Assign datapoints to closest centroids
  for (int m = startM; m < endM; m++)
  {
    // initialize arrays
    minDist[m] = 1e30;
    //args->clusterAssignments[m] = -1;
    for (int k = startK; k < endK; k++)
    {
      double d = dist(&data[m * N], &clusterCentroids[k * N], N);
      if (d < minDist[m])
      {
        minDist[m] = d;
        clusterAssignments[m] = k;
      }
    }
  }
}

/**
 * Given the cluster assignments, computes the new centroid locations for
 * each cluster.
 */
void computeCentroids_bythread(WorkerArgs *const args) {
  int startM = args->startM;
  int endM = args->endM;
  int N = args->N;
  int K = args->K;
  double *data = args->data;
  int *clusterAssignments = args->clusterAssignments;
  double *thread_centroids = args->thread_centroids;

  // Sum up contributions from assigned examples
  __m256d vec_x = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);
  __m256d vec_y = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);
  __m256d vec_z = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);
  for (int m = startM; m < endM; m++)
  {
    int k = clusterAssignments[m];
    for (int n = 0; n < N; n+=4) {
      int index = k * N + n;
      vec_x = _mm256_loadu_pd(thread_centroids + index);
      vec_y = _mm256_loadu_pd(data + (m * N + n));
      _mm256_storeu_pd(thread_centroids + index, _mm256_add_pd(vec_x, vec_y));
    }
    args->thread_counts[k]++;
  }
}

/**
 * Computes the per-cluster cost. Used to check if the algorithm has converged.
 */
void computeCost(WorkerArgs *const args) {
  int startM = args->startM;
  int endM = args->endM;
  int M = args->M;
  int N = args->N;
  double *data = args->data;
  int *clusterAssignments = args->clusterAssignments;
  double *clusterCentroids = args->clusterCentroids;
  double *thread_accum = args->thread_accum;
  // Sum cost for all data points assigned to centroid
  for (int m = startM; m < endM; m++) {
    int k = clusterAssignments[m];
    thread_accum[k] += dist(&data[m * N], &clusterCentroids[k * N], N);
  }
}

/**
 * Computes the K-Means algorithm, using std::thread to parallelize the work.
 *
 * @param data Pointer to an array of length M*N representing the M different N 
 *     dimensional data points clustered. The data is layed out in a "data point
 *     major" format, so that data[i*N] is the start of the i'th data point in 
 *     the array. The N values of the i'th datapoint are the N values in the 
 *     range data[i*N] to data[(i+1) * N].
 * @param clusterCentroids Pointer to an array of length K*N representing the K 
 *     different N dimensional cluster centroids. The data is laid out in
 *     the same way as explained above for data.
 * @param clusterAssignments Pointer to an array of length M representing the
 *     cluster assignments of each data point, where clusterAssignments[i] = j
 *     indicates that data point i is closest to cluster centroid j.
 * @param M The number of data points to cluster.
 * @param N The dimensionality of the data points.
 * @param K The number of cluster centroids.
 * @param epsilon The algorithm is said to have converged when
 *     |currCost[i] - prevCost[i]| < epsilon for all i where i = 0, 1, ..., K-1
 */
void compute(WorkerArgs *const args){
  double startTime1 = CycleTimer::currentSeconds();
  double *minDist = new double[args->M](); // for computeAssignments_bythread
  // Create thread objects
  std::thread workers[MAX_THREADS];
  WorkerArgs threadArgs[MAX_THREADS];
  int step = args->M / MAX_THREADS;
  for (int i = 0; i < MAX_THREADS; i++)
  {
    threadArgs[i].data = args->data;
    threadArgs[i].clusterCentroids = args->clusterCentroids;
    threadArgs[i].clusterAssignments = args->clusterAssignments;
    threadArgs[i].currCost = args->currCost;
    threadArgs[i].M = args->M;
    threadArgs[i].N = args->N;
    threadArgs[i].K = args->K;

    threadArgs[i].startM = step * i;
    threadArgs[i].endM = step * (i + 1);
    threadArgs[i].startK = args->startK;
    threadArgs[i].endK = args->endK;
    threadArgs[i].step = step;
    threadArgs[i].threadID = i;
    ////////////////////////////
    threadArgs[i].temp = minDist; // for computeAssignments_bythread
    threadArgs[i].thread_centroids = new double[args->K * args->N](); // for computeCentroids
    threadArgs[i].thread_counts = new int[args->K]();  
    threadArgs[i].thread_accum = new double[args->K](); // for computeCost
  }

  for (int i = 1; i < MAX_THREADS; i++)
  {
    // Create threads
    workers[i] = std::thread(computeAssignments_bythread, &threadArgs[i]);
  }
  // Call the function for the first thread
  computeAssignments_bythread(&threadArgs[0]);
  for (int i = 1; i < MAX_THREADS; i++)
  {
    // Join threads
    workers[i].join();
  }
  double endTime1 = CycleTimer::currentSeconds();
  printf("[Total Time1]: %.3f ms\n", (endTime1 - startTime1) * 1000);

  double startTime2 = CycleTimer::currentSeconds();
  // Zero things out

  for (int i = 1; i < MAX_THREADS; i++)
  {
    // Create threads
    workers[i] = std::thread(computeCentroids_bythread, &threadArgs[i]);
  }
  // Call the function for the first thread
  computeCentroids_bythread(&threadArgs[0]);
  
  for (int i = 1; i < MAX_THREADS; i++)
  {
    // Join threads
    workers[i].join();
  }
  for (int i = 1; i < MAX_THREADS; i++) {
    for (int k = 0; k < args->K; k++)
    {
      for (int n = 0; n < args->N; n++)
      {
        int index = k * args->N + n;
        threadArgs[0].thread_centroids[index] += threadArgs[i].thread_centroids[index];
      }
      threadArgs[0].thread_counts[k] += threadArgs[i].thread_counts[k];
    }
  }
  
  for (int k = 0; k < args->K; k++)
  {
    threadArgs[0].thread_counts[k] = max(threadArgs[0].thread_counts[k], 1); // prevent divide by 0
    for (int n = 0; n < args->N; n++)
    {
      threadArgs[0].clusterCentroids[k * args->N + n] = threadArgs[0].thread_centroids[k * args->N + n] / threadArgs[0].thread_counts[k];
    }
  }
  double endTime2 = CycleTimer::currentSeconds();
  printf("[Total Time2]: %.3f ms\n", (endTime2 - startTime2) * 1000);

  double startTime3 = CycleTimer::currentSeconds();
  for (int i = 1; i < MAX_THREADS; i++)
  {
    // Create threads
    workers[i] = std::thread(computeCost, &threadArgs[i]);
  }
  computeCost(&threadArgs[0]); // Update costs
  for (int i = 1; i < MAX_THREADS; i++)
  {
    // Join threads
    workers[i].join();
  }
  for (int i = 1; i < MAX_THREADS; i++)
  {
    for (int k = 0; k < args->K; k++)
    {
      threadArgs[0].thread_accum[k] += threadArgs[i].thread_accum[k];
    }
  }
  for (int k = args->startK; k < args->endK; k++)
  {
    args->currCost[k] = threadArgs[0].thread_accum[k];
  }
  double endTime3 = CycleTimer::currentSeconds();
  printf("[Total Time3]: %.3f ms\n", (endTime3 - startTime3) * 1000);
  free(minDist);
  for (int i = 0; i < MAX_THREADS; i++)
  {
    delete[] threadArgs[i].thread_centroids;
    delete[] threadArgs[i].thread_counts;
    delete[] threadArgs[i].thread_accum;
  }

}
void kMeansThread(double *data, double *clusterCentroids, int *clusterAssignments,
               int M, int N, int K, double epsilon) {

  // Used to track convergence
  double *prevCost = new double[K];
  double *currCost = new double[K];

  // The WorkerArgs array is used to pass inputs to and return output from
  // functions.
  WorkerArgs args{
      .M = M,
      .N = N,
      .K = K,
      .data = data,
      .clusterCentroids = clusterCentroids,
      .clusterAssignments = clusterAssignments,
      .currCost = currCost,
  };

  // Initialize arrays to track cost
  for (int k = 0; k < K; k++) {
    prevCost[k] = 1e30;
    currCost[k] = 0.0;
  }

  /* Main K-Means Algorithm Loop */
  int iter = 0;
  while (!stoppingConditionMet(prevCost, currCost, epsilon, K)) {
    // Update cost arrays (for checking convergence criteria)
    printf("this is the %d iteration\n", iter + 1);
    for (int k = 0; k < K; k++) {
      prevCost[k] = currCost[k];
    }

    // Setup args struct
    args.startK = 0;
    args.endK = K;
    compute(&args);

    iter++;
  }
  printf("K-Means converged after %d iterations.\n", iter);
  free(currCost);
  free(prevCost);
}
