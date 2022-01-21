#ifndef _SOFT_MUTUAL_SCORE_CUH_ON
#define _SOFT_MUTUAL_SCORE_CUH_ON

void soft_mutual_score(float* temp_score0, int* temp_idx0, 
                        float* temp_score1, int* temp_idx1, 
                        int* idx0, int * idx1, 
                        const int n, const int* pred_pair, 
                        const float* reg_feat, float* score);
void soft_mutual_score_grad(const float* temp_score0, const int* temp_idx0, 
    const float* temp_score1, const int* temp_idx1, 
    const int* idx0, const int * idx1, 
    const int n, const float* graddist, float* grad_reg_feat);

#endif