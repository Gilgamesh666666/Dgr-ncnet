#include<stdio.h>
#include"../cuda_utils.cuh"

//float[N0,1], int[N1,1], float[N0,1], int[N1,1], 
//int[n,1], int[n,1], int[n,2], int[n,1]
__global__ void soft_mutual_score_kernel(float *__restrict__ temp_score0, int *__restrict__ temp_idx0,
    float *__restrict__ temp_score1, int *__restrict__ temp_idx1, 
    int *__restrict__ idx0, int *__restrict__ idx1, 
    const int n, const int *__restrict__ pred_pair, 
    const float *__restrict__ reg_feat)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x*gridDim.x;

    for(int i=index;i<n;i+=stride)
    {
        int id0 = pred_pair[i*2];
        int id1 = pred_pair[i*2 + 1];
        idx0[i] = id0;
        idx1[i] = id1;
        float feat = reg_feat[i];
        float max_feat0 = temp_score0[id0];
        float max_feat1 = temp_score1[id1];
        __syncthreads();
        if(feat>max_feat0)
        {
            temp_score0[id0] = reg_feat[i];
            temp_idx0[id0] = i;
        };
        __syncthreads();
        if(feat>max_feat1)
        {
            temp_score1[id1] = reg_feat[i];
            temp_idx1[id1] = i;
        };
    };
};
//float[N0,1], int[n,1], float[N0,1], int[n,1], 
__global__ void fill_in_result(const int n, const float *__restrict__ temp_score0, 
    const float *__restrict__ temp_score1,const int *__restrict__ idx0, 
    const int *__restrict__ idx1, float *__restrict__ score)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x*gridDim.x;

    for(int i=index;i<n;i+=stride)
    {
        int id0 = idx0[i];
        int id1 = idx1[i];
        score[i] = temp_score0[id0] * temp_score1[id1];
    };
}
//ouput [n,1]
__global__ void soft_mutual_score_grad_kernel(const float *__restrict__ temp_score0, const int *__restrict__ temp_idx0,
    const float *__restrict__ temp_score1, const int *__restrict__ temp_idx1, 
    const int *__restrict__ idx0, const int *__restrict__ idx1,
    const int n, const float *__restrict__ graddist, float *__restrict__ grad_reg_feat)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x*gridDim.x;
    for(int i=index;i<n;i+=stride)
    {
        int id0 = idx0[i];
        int id1 = idx1[i];
        int max_idx0 = temp_idx0[id0];
        float max_score0 = temp_score0[id0];
        int max_idx1 = temp_idx1[id1];
        float max_score1 = temp_score1[id1];
        float g = graddist[i];
        atomicAdd(grad_reg_feat + max_idx0, g*max_score1);
        atomicAdd(grad_reg_feat + max_idx1, g*max_score0);
    }
};
void soft_mutual_score(float* temp_score0, int* temp_idx0, 
                        float* temp_score1, int* temp_idx1,
                        int* idx0, int * idx1, 
                        const int n, const int* pred_pair, 
                        const float* reg_feat, float* score)
{
    int thead_num = optimal_num_threads(n);
    soft_mutual_score_kernel<<<max(int(n/thead_num), 1), thead_num>>>(temp_score0, temp_idx0, 
                                                                    temp_score1, temp_idx1,
                                                                    idx0, idx1,
                                                                    n, pred_pair, reg_feat);
    CUDA_CHECK_ERRORS();
    cudaThreadSynchronize();
    fill_in_result<<<max(int(n/thead_num), 1), thead_num>>>(n, temp_score0, temp_score1,idx0, idx1,score);
    CUDA_CHECK_ERRORS();
};

void soft_mutual_score_grad(const float* temp_score0, const int* temp_idx0, 
    const float* temp_score1, const int* temp_idx1,
    const int* idx0, const int * idx1, 
    const int n, const float* graddist, float* grad_reg_feat)
{
    int thead_num = optimal_num_threads(n);
    soft_mutual_score_grad_kernel<<<max(int(n/thead_num), 1), thead_num>>>(temp_score0, temp_idx0, 
                                                                            temp_score1, temp_idx1,
                                                                            idx0, idx1,
                                                                           n, graddist, grad_reg_feat);
    CUDA_CHECK_ERRORS();
};
    