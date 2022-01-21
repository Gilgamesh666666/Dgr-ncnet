# functions/add.py
import torch
from extension.functional.backend import _backend
__all__ = ['nearest_neighbor']

class Soft_Mutual_Score_On(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred_pair, reg_feat):
        # one point pair
        # [n3, 2], [n3, 1]
        # [n3, 1]
        # pred_pair, reg_feat
        reg_feat = reg_feat.float().contiguous()
        pred_pair = pred_pair.int().contiguous()
        temp = torch.max(pred_pair, dim=0).values.long()
        N0,N1 = temp[0]+1, temp[1]+1
        temp_score0, temp_idx0, temp_score1, temp_idx1, idx0, idx1, score = _backend.soft_mutual_score_forward_On(N0, N1, pred_pair, reg_feat)
        ctx.mark_non_differentiable(pred_pair, temp_score0, temp_idx0, temp_score1, temp_idx1, idx0, idx1)
        ctx.save_for_backward(temp_score0, temp_idx0, temp_score1, temp_idx1, idx0, idx1)
        return score
    @staticmethod
    def backward(ctx, graddist):
        temp_score0, temp_idx0, temp_score1, temp_idx1, idx0, idx1 = ctx.saved_tensors
        graddist = graddist.contiguous()
        #print(temp_score0, temp_idx0, temp_score1, temp_idx1)
        grad_reg_feat = _backend.soft_mutual_score_backward_On(temp_score0, temp_idx0, temp_score1, temp_idx1, idx0, idx1, graddist)
        #print(grad_reg_feat.shape)
        return None, grad_reg_feat

soft_mutual_score_On = Soft_Mutual_Score_On.apply