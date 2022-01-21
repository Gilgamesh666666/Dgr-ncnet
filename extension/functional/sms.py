# functions/add.py
import torch
from extension.functional.backend import _backend
__all__ = ['nearest_neighbor']

class Soft_Mutual_Score(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred_pair, reg_feat):
        # one point pair
        # [n3, 2], [n3, 1]
        # [n3, 1]
        # pred_pair, reg_feat
        reg_feat = reg_feat.float().contiguous()
        pred_pair = pred_pair.int().contiguous()
        temp_score0, temp_idx0, temp_score1, temp_idx1, score = _backend.soft_mutual_score_forward(pred_pair, reg_feat)
        ctx.mark_non_differentiable(pred_pair, temp_score0, temp_idx0, temp_score1, temp_idx1)
        ctx.save_for_backward(temp_score0, temp_idx0, temp_score1, temp_idx1)
        return score
    @staticmethod
    def backward(ctx, graddist):
        temp_score0, temp_idx0, temp_score1, temp_idx1 = ctx.saved_tensors
        graddist = graddist.contiguous()
        #print(temp_score0, temp_idx0, temp_score1, temp_idx1)
        grad_reg_feat = _backend.soft_mutual_score_backward(temp_score0, temp_idx0, temp_score1, temp_idx1, graddist)
        #print(grad_reg_feat.shape)
        return None, grad_reg_feat

soft_mutual_score = Soft_Mutual_Score.apply