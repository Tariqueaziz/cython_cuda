#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include "kmeans.h"


#define sharedMemSize 2000

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err){
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


//class kmeans{
//public:
//	int* getClusters(float *h_data, int n, int dim, int nc, int iter);
//};


__global__ void updateCluster(int *d_clusterid, float *d_data, float *d_centroids, int n, int nc, int dim){
	const long unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= n) return;
	int index = 0;
	float distance = FLT_MAX;
	float d;
    for(int i = 0; i < nc; i++){
    	d = 0;
		for(int k = 0; k < dim; k++){
			d += (d_data[n*k + id] - d_centroids[nc*k + i]) * (d_data[n*k + id] - d_centroids[nc*k + i]);
		}


    	if(d < distance){
    		distance = d;
    		index = i;
    	}
    }
    d_clusterid[id] = index;
}

__global__ void updateCentroids(float *d_data, int *d_clusterid, float *d_centroids, int *d_clustercount,
		const int n, const int dim, const int nc){

	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= n) return;

	const int s_id = threadIdx.x;

	__shared__ float s_data[sharedMemSize];
	for(int i = 0; i < dim; i++){
		s_data[i * blockDim.x + s_id]= d_data[n * i + id];
	}
	__shared__ int s_clusterid[sharedMemSize];
	s_clusterid[s_id] = d_clusterid[id];
	__syncthreads();

	if(s_id == 0)
	{
		float accumulate_sums[sharedMemSize]={0};
		int accumulate_clustersize[sharedMemSize]={0};

		for(int j = 0; j < blockDim.x and blockIdx.x * blockDim.x + j < n; j++)
		{
			int clust_id = s_clusterid[j];

			accumulate_clustersize[clust_id]+=1;

			for(int k = 0; k < dim; k++){
				accumulate_sums[nc * k + clust_id] += s_data[blockDim.x * k + j];
			}

		}
		for(int j=0; j < nc; j++){
			for(int k = 0; k < dim; k++){

				atomicAdd(&d_centroids[nc * k + j], accumulate_sums[nc * k + j]);
			}
			atomicAdd(&d_clustercount[j], accumulate_clustersize[j]);
		}
	}

	__syncthreads();

}

__global__ void scaleCentroids(float *d_centroids, int *d_clustercount, const int nc, const int dim){
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= nc) return;

	for(int i = 0; i < dim; i++){
		d_centroids[nc * i + id] /= d_clustercount[id];
	}
}



void getClustersH(int *clusters, float *h_data, int n, int dim, int nc, int iter){
	int numThreads = 32;
	int numBlocks = (n-1)/numThreads + 1;
	thrust::host_vector<float> h_data_vector(h_data, h_data + n * dim);
	thrust::device_vector<float> d_data(n * dim, 0);
	d_data = h_data_vector;

	thrust::host_vector<float> h_centroids;
	srand(1000);
	for(int i = 0; i < dim; i++){
		for(int j = 0; j < nc; j++){

			float val = (rand()%100) / 100.0;
			h_centroids.push_back(val);
		}
	}

	thrust::device_vector<float> d_centroids(nc * dim, 0);
	d_centroids = h_centroids;

	thrust::device_vector<int> d_clustercount(nc, 0);

	thrust::device_vector<int> d_clusterid(n, 0);


	for(int i = 0; i < iter; i++){
		updateCluster<<<numBlocks, numThreads>>>((int*)thrust::raw_pointer_cast(d_clusterid.data()),
				(float*)thrust::raw_pointer_cast(d_data.data()),
				(float*)thrust::raw_pointer_cast(d_centroids.data()), n, nc, dim);
		CUDA_CHECK_RETURN(cudaThreadSynchronize());
		thrust::fill(d_centroids.begin(), d_centroids.end(), 0);
		thrust::fill(d_clustercount.begin(), d_clustercount.end() + nc, 0);
		updateCentroids<<<numBlocks, numThreads>>>((float*)thrust::raw_pointer_cast(d_data.data()), (int*)thrust::raw_pointer_cast(d_clusterid.data()),
				(float*)thrust::raw_pointer_cast(d_centroids.data()), (int*)thrust::raw_pointer_cast(d_clustercount.data()),
				n, dim, nc);

		CUDA_CHECK_RETURN(cudaThreadSynchronize());


		scaleCentroids<<<(nc-1)/numThreads + 1, numThreads>>>((float*)thrust::raw_pointer_cast(d_centroids.data()),
				(int*)thrust::raw_pointer_cast(d_clustercount.data()), nc, dim);

	}
//	int* raw_h_clusterid = (int*) malloc(n * sizeof(int));

	int* raw_d_clusterid = (int*)thrust::raw_pointer_cast(d_clusterid.data());

	CUDA_CHECK_RETURN(cudaMemcpy(clusters, raw_d_clusterid, n * sizeof(int), cudaMemcpyDeviceToHost));

//	CUDA_CHECK_RETURN(cudaDeviceReset());
//	return raw_h_clusterid;
}


void kmeans::getClusters(int *clusters, float *h_data, int n, int dim, int nc, int iter){
    getClustersH(clusters, h_data, n, dim, nc, iter);
    CUDA_CHECK_RETURN(cudaDeviceReset());
}