#include<stdio.h>
#include"../cuda_utils.cuh"

//float[n,1], int[n,1], float[n,1], int[n,1], 
//float[n,2], int[n,1], int[n,1]
__global__ void soft_mutual_score_kernel(float *__restrict__ temp_score0, int *__restrict__ temp_idx0,
    float *__restrict__ temp_score1, int *__restrict__ temp_idx1, 
    const int n, const int *__restrict__ pred_pair, 
    const float *__restrict__ reg_feat)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x*gridDim.x;

    for(int i=index;i<n;i+=stride)
    {
        int id0 = pred_pair[i*2];
        int id1 = pred_pair[i*2 + 1];
        for(int j=0;j<n;j++)
        {
            // max_ijab
            if(pred_pair[j*2]==id0)
            {
                if (reg_feat[j]>temp_score0[i])
                {
                    temp_score0[i] = reg_feat[j];
                    temp_idx0[i] = j;
                };
            };
            // max_cdkl
            if(pred_pair[j*2+1]==id1)
            {
                if (reg_feat[j]>temp_score1[i])
                {
                    temp_score1[i] = reg_feat[j];
                    temp_idx1[i] = j;
                };
            };
        };
    };
};
__global__ void fill_in_result(const int n, const float *__restrict__ temp_score0, 
    const float *__restrict__ temp_score1,float *__restrict__ score)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x*gridDim.x;

    for(int i=index;i<n;i+=stride)
    {
        score[i] = temp_score0[i] * temp_score1[i];
    };
}
//ouput [n,1]
__global__ void soft_mutual_score_grad_kernel(const float *__restrict__ temp_score0, const int *__restrict__ temp_idx0,
    const float *__restrict__ temp_score1, const int *__restrict__ temp_idx1, 
    const int n, const float *__restrict__ graddist, float *__restrict__ grad_reg_feat)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x*gridDim.x;
    for(int i=index;i<n;i+=stride)
    {
        int idx0 = temp_idx0[i];
        float score0 = temp_score0[i];
        int idx1 = temp_idx1[i];
        float score1 = temp_score1[i];
        float g = graddist[i];
        atomicAdd(grad_reg_feat + idx0, g*score1);
        atomicAdd(grad_reg_feat + idx1, g*score0);
    }
};
void soft_mutual_score(float* temp_score0, int* temp_idx0, 
                        float* temp_score1, int* temp_idx1, 
                        const int n, const int* pred_pair, 
                        const float* reg_feat, float* score)
{
    int thead_num = optimal_num_threads(n);
    soft_mutual_score_kernel<<<max(int(n/thead_num), 1), thead_num>>>(temp_score0, temp_idx0, 
                                                                    temp_score1, temp_idx1, 
                                                                    n, pred_pair, reg_feat);
    CUDA_CHECK_ERRORS();
    cudaThreadSynchronize();
    fill_in_result<<<max(int(n/thead_num), 1), thead_num>>>(n, temp_score0, temp_score1,score);
    CUDA_CHECK_ERRORS();
};

void soft_mutual_score_grad(const float* temp_score0, const int* temp_idx0, 
    const float* temp_score1, const int* temp_idx1, 
    const int n, const float* graddist, float* grad_reg_feat)
{
    int thead_num = optimal_num_threads(n);
    soft_mutual_score_grad_kernel<<<max(int(n/thead_num), 1), thead_num>>>(temp_score0, temp_idx0, 
                                                                            temp_score1, temp_idx1, 
                                                                           n, graddist, grad_reg_feat);
    CUDA_CHECK_ERRORS();
};
    