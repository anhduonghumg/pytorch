import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# Tensor giống kiểu ndarrays của NumPy nhưng có thể tận dụng được năng lục GPU
#*** Một ma trận được khai báo nhưng chưa được khởi tạo sẽ chứa các giá trị ngẫu nhiên ứng với vùng nhớ được cấp phát

#. Một ma trận 5x3 chưa được khởi tạo
# x = torch.empty(5,3)
# print(x)

#. Tạo và khỏi tạo 1 ma trận ngẫu nhiên
# y = torch.rand(5,3)
# print(y)

#. Tạo ma trận toàn 0 với kiểu long
# long = torch.zeros(5,3, dtype=torch.long)
# print(long)

# tạo 1 tensor trực tiếp từ dữ liệu
tensor = torch.tensor([5.5,3])
# print(tensor)

# tạo 1 tensor từ một tensor có trước đó,Phương thức này 
one = tensor.new_ones(5,3,dtype=torch.double)
like = torch.randn_like(tensor, dtype=torch.float)

# lây thông tin kích thước tensor
# print(one.size())

# CAC PHEP TINH

# Cong
x = torch.rand(5,3)
y = torch.rand(5,3)

# cong = torch.add(x,y)
# print(cong)

# Phep cong nhu 1 doi so
# result = torch.empty(5,3)
# congDS = torch.add(x,y, out=result)
# print(result)

# phep cong dang in-place
# y.add_(x)
# print(y)

# dinh dang lại 1 tensor = view
# view = x.view(15)
# print(view)

# Chuyen doi voi Numpy
# a = np.ones(5)
# b = torch.from_numpy(a)

# np.add(a,1, out=a)
# a.add_(1)
# print(a)
# print(b)

# CUDA TENSOR 

# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     y = torch.ones_like(x, device=device)
#     x = x.to(device)
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))
# print('not ok')