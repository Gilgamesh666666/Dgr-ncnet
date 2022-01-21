#include<vector>
#include "soft_mutual_score.hpp"
#include "soft_mutual_score.cuh"
#include "../utils.hpp"

std::vector<at::Tensor> soft_mutual_score_forward(at::Tensor pred_pair, at::Tensor reg_feat)
{
    CHECK_INPUT(pred_pair);CHECK_IS_INT(pred_pair);
    CHECK_INPUT(reg_feat);CHECK_IS_FLOAT(reg_feat);
    int n = pred_pair.size(0);
    at::Tensor score = torch::zeros_like(reg_feat, at::device(reg_feat.device()).dtype(at::ScalarType::Float));
    at::Tensor temp_score0 = torch::zeros_like(reg_feat, at::device(reg_feat.device()).dtype(at::ScalarType::Float));
    at::Tensor temp_score1 = torch::zeros_like(reg_feat, at::device(reg_feat.device()).dtype(at::ScalarType::Float));
    at::Tensor temp_idx0 = torch::zeros_like(reg_feat, at::device(reg_feat.device()).dtype(at::ScalarType::Int));
    at::Tensor temp_idx1 = torch::zeros_like(reg_feat, at::device(reg_feat.device()).dtype(at::ScalarType::Int));
    soft_mutual_score(temp_score0.data_ptr<float>(), temp_idx0.data_ptr<int>(), 
    temp_score1.data_ptr<float>(), temp_idx1.data_ptr<int>(), 
    n, pred_pair.data_ptr<int>(), reg_feat.data_ptr<float>(),
    score.data_ptr<float>());
    
    return {temp_score0, temp_idx0, temp_score1, temp_idx1, score};
};
at::Tensor soft_mutual_score_backward(at::Tensor temp_score0, at::Tensor temp_idx0, 
                                    at::Tensor temp_score1, at::Tensor temp_idx1, 
                                    at::Tensor graddist)
{
    CHECK_INPUT(temp_score0);CHECK_IS_FLOAT(temp_score0);
    CHECK_INPUT(temp_idx0);CHECK_IS_INT(temp_idx0);
    CHECK_INPUT(temp_score1);CHECK_IS_FLOAT(temp_score1);
    CHECK_INPUT(temp_idx1);CHECK_IS_INT(temp_idx1);
    CHECK_INPUT(graddist);CHECK_IS_FLOAT(graddist);
    int n = temp_score0.size(0);
    at::Tensor grad_reg_feat = torch::zeros_like(temp_score0, at::device(graddist.device()).dtype(at::ScalarType::Float));
    soft_mutual_score_grad(temp_score0.data_ptr<float>(), temp_idx0.data_ptr<int>(), 
                            temp_score1.data_ptr<float>(), temp_idx1.data_ptr<int>(), 
                            n, graddist.data_ptr<float>(), grad_reg_feat.data_ptr<float>());
    return grad_reg_feat;
};