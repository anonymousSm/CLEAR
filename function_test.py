import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np

a = [12, 11, 12, 13, 13, 23, 25, 25, 27]
b = {}
c = []
index = 1
for i in a:
    if i in b:
        continue
    else:
        b[i] = index
        index += 1

print(b)

for j in a:
    c.append(b.get(j))

print(np.array(c))