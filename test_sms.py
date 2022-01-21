import torch
import time
from extension.sms import soft_mutual_score_Module
from extension.sms_On import soft_mutual_score_Module_On
n=10000
pred_pair = (torch.rand((n, 2))*torch.tensor([6000, 10000])).int().cuda() #[b, c, n]
reg_feat = torch.rand(n).cuda().requires_grad_() #[b, c, m]

class gtsms(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred_pair ,reg_feat):
        with torch.no_grad():
            output0, inverse_indices0 = torch.unique(pred_pair[:, 0], return_inverse=True)
            output1, inverse_indices1 = torch.unique(pred_pair[:, 1], return_inverse=True)

        temp = torch.zeros_like(reg_feat).to(reg_feat)
        soft_mutual_score = torch.zeros_like(reg_feat).to(reg_feat)
        for i in range(len(output0)):
            #max_c_ijab
            temp[inverse_indices0==i] = reg_feat[inverse_indices0==i].max().item()

        for i in range(len(output1)):
            #max_c_cdkl
            soft_mutual_score[inverse_indices1==i] = temp[inverse_indices1==i]*reg_feat[inverse_indices1==i].max().item()
        return soft_mutual_score

class gtsms_fast(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred_pair ,reg_feat):
        soft_mutual_score = reg_feat
        temp = pred_pair.max(dim=0).values.long()
        N0, N1 = temp[0].item(), temp[1].item()
        max_score0 = torch.zeros(N0+1)
        max_score1 = torch.zeros(N1+1)
        for i, pair_idx in enumerate(pred_pair):
            id0, id1 = pair_idx
            if reg_feat[i] > max_score0[id0]:
                max_score0[id0] = reg_feat[i]
            if reg_feat[i] > max_score1[id1]:
                max_score1[id1] = reg_feat[i]
        #max_c_ijab, max_c_cdkl
        for i, pair_idx in enumerate(pred_pair):
            soft_mutual_score[i] = max_score0[pair_idx[0]] * max_score1[pair_idx[1]]
        return soft_mutual_score
stime = time.time()
dist = soft_mutual_score_Module()
print(dist(pred_pair, reg_feat))
sms_cuda_time = time.time() - stime

stime = time.time()
gt = gtsms()
print(gt(pred_pair, reg_feat))
sms_time = time.time() - stime

stime = time.time()
gt_fast = gtsms_fast()
print(gt_fast(pred_pair, reg_feat))
sms_On_time = time.time() - stime

print(sms_cuda_time, sms_time, sms_On_time)
#print(torch.autograd.gradcheck(dist, (pred_pair ,reg_feat), eps=1e-3))