#ifndef _SOFT_MUTUAL_SCORE_HPP_ON
#define _SOFT_MUTUAL_SCORE_HPP_ON

#include<torch/extension.h>
#include<vector>

std::vector<at::Tensor> soft_mutual_score_forward_On(int N0, int N1, at::Tensor pred_pair, at::Tensor reg_feat);
at::Tensor soft_mutual_score_backward_On(at::Tensor temp_score0, at::Tensor temp_idx0, 
                                    at::Tensor temp_score1, at::Tensor temp_idx1, 
                                    at::Tensor idx0, at::Tensor idx1,
                                    at::Tensor graddist);
#endif