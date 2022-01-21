import torch
from extension.knn import knnModule

data1 = torch.rand((2, 3, 5)).cuda().requires_grad_() #[b, c, n]
data2 = torch.rand((2, 3, 4)).cuda().requires_grad_() #[b, c, m]
#print(f'data1.is_leaf={data1.is_leaf}')
class gtknn(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a ,b, k):
        compare = (a.unsqueeze(3) - b.unsqueeze(2)).norm(dim=1) #[b,n,m]
        dist1, idx1 = torch.topk(compare, dim=2, k=k, largest=False)
        dist2, idx2 = torch.topk(compare, dim=1, k=k, largest=False)
        return dist1.permute(0, 2, 1), dist2, idx1.permute(0, 2, 1), idx2

dist = knnModule()
gtdist = gtknn()
k=3
dist1, dist2, idx1, idx2 = gtdist(data1, data2, k)
#print(f'gt:dist1={dist1}\ndist2={dist2}\nidx1={idx1}\nidx2={idx2}')
dist1sum = dist1.sum()
dist1sum.backward()
print(data1.grad)
data1.grad.data.zero_()
dist1, dist2, idx1, idx2 = dist(data1, data2, k)
dist1sum = dist1.sum()
dist1sum.backward()
print(data1.grad)
#print(f'dist1={dist1}\ndist2={dist2}\nidx1={idx1}\nidx2={idx2}')
#print(torch.autograd.gradcheck(dist, (data1, data2, k), eps=1e-3))