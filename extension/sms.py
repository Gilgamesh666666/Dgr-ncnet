import torch.nn as nn
from extension.functional.sms import soft_mutual_score
import torch
class soft_mutual_score_Module(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred_pair, reg_feat):
        return soft_mutual_score(pred_pair, reg_feat)