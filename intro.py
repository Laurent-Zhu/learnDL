import torch

# x = torch.arange(12)
# print(x)
# print(x.shape)
# print(x.numel())
# X = x.reshape(-1, 4)
# print(X)

# # y = torch.zeros((2,3,4))
# # print(y.numel())
# # print(y)
# y = torch.tensor([[2,4,3], [3,4,5],[3,5,6]])
# # print(y.sum())
# # print(y[-1], y[1:3], y[1,2])
# # y[0:2, :] = 6
# # print(y)
# before = id(y)
# y += y
# print(id(y) == before)
# import os

# os.makedirs(os.path.join('..', 'data'), exist_ok=True)
# datafile = os.path.join('..', 'data', 'houst_tiny.csv')
# with open(datafile, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')
#     f.write('NA,Pave,127500\n')
#     f.write('4,NA,106000')

# import pandas as pd
# data = pd.read_csv(datafile)
# print(data)

# from plotutils import use_svg_display, set_figsize, set_axes, plot
# import numpy as np
# import matplotlib.pyplot as plt

# def f(x):
#     return 3*x**2 - 4*x

# x = np.arange(0, 3, 0.1)
# plot(x, [f(x), 2*x-3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
# plt.show()

x = torch.arange(4.0, requires_grad=True)
y = 2 * torch.dot(x, x)
y.backward()
print(x.grad)
print(x.grad == 4 * x)

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

