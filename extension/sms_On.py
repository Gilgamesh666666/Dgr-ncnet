import torch.nn as nn
from extension.functional.sms_On import soft_mutual_score_On
import torch
class soft_mutual_score_Module_On(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred_pair, reg_feat):
        return soft_mutual_score_On(pred_pair, reg_feat)