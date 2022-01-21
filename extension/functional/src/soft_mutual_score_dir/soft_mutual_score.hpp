#ifndef _SOFT_MUTUAL_SCORE_HPP
#define _SOFT_MUTUAL_SCORE_HPP

#include<torch/extension.h>
#include<vector>

std::vector<at::Tensor> soft_mutual_score_forward(at::Tensor pred_pair, at::Tensor reg_feat);
at::Tensor soft_mutual_score_backward(at::Tensor temp_score0, at::Tensor temp_idx0, 
                                    at::Tensor temp_score1, at::Tensor temp_idx1, 
                                    at::Tensor graddist);
#endif