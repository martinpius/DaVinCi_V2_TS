import torch , math
E = torch.nn.Embedding(8, 100)
E1 = torch.randn(size = (8, 100))
p = torch.arange(8, dtype = torch.long)
e = E(p)
e1 = p.unsqueeze_(dim = 1).repeat(1, 8).float() @ E1
assert e.shape == (8, 100) == e1.shape

A = []
for _ in range(10):
    A.append(torch.randn(2, 10))
R = torch.stack(A, dim = 0)
assert R.shape == (10, 2, 10)

def cosine_lr(it, 
              max_lr = 6e-4,
              max_iters = 1000,
              warm_up = 100):
    
    min_lr = 0.1 * max_lr
    if it < warm_up:
        lr = max_lr * (it + 1) / (warm_up)
    elif it > max_iters:
        lr = min_lr 
    else:
        rate = (it - warm_up) / (max_iters - warm_up)
        rate = 0.5 * (1.0 + math.cos(rate * math.pi))
        lr = min_lr + rate * (max_lr - min_lr)
    return lr 
import matplotlib.pyplot as plt 
lrs = []
for it in range(1000):
    lr = cosine_lr(it)
    lrs.append(lr)
plt.plot(lrs)
plt.show()
