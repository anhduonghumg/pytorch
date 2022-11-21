# Autograd: Tự động tính vi phân

#. Cốt lõi của các mạng nơ ron trong Pytorch là autoGrad
#. Gói autoGrad cung cấp phép tính vi phân tự động cho phép toán trên Tensor

# Tensor 
# torch.tensor if .requires_grad được đặt bằng True, autograd sẽ theo dõi mọi phép toán thực hiện trên tensor đó
# Khi đã thực hiện xong, ta có thể gọi phương thức .backward và các giá trị đạo hàm được tính 1 cách tự động, gt này đc cộng dồn vafp .grad của tensor
# để dừng theo dõi 1 tensor ta dùng .detach để tách nó ra khỏi lịch sử tính toán và các phép toán sau này cx ko đc theo dõi
# Có thể sử dụng khối with torch.no_grad(): hữu ích khi ta đánh giá mô hình vì 1 mô hình có thể chứa các tham số với requires_grad = true những đánh giá ta ko qt đến đạo hàm
# 1 lớp quan trong khác la Function (hàm)
# Tensor va Function kết nối với nhau thành một đồ thị không có chu trình biểu diễn lịch sử tính toán
# -- mỗi Tensor có 1 thuộc tính .grand_fn trở đến 1 Function tạo ra nó (Ngoại trừ các Tensor được tạo bởi người dùng, thuộc tính grand_fn của chúng là none)
# Nếu muốn tính đạo hàm ta có thể gọi .backward() trên 1 tensor
# --- if Tensor là kiểu số (scalar - tức chỉ chứa 1 pt) ta ko cần chỉnh đối số cho backward()
# --- if có nhiều hơn 1 pt ta pải chỉ rõ đối số grandient với định dạng phù hợp

# các hàm sử dụng
# -- requires_grand (true || false)
# -- backward
# -- detach
# -- function
# -- grand_fn

import torch

# tạo 1 tensor và đặt requires_grad = true để theo dõi phép toán trên nó
x = torch.ones(2,2, requires_grad=True)
y = x + 2

# y được tạo từ kết quả của 1 phép tính toán nên nó có thuộc tính grad_fn
# print (y.grad_fn)

# thực hiện thêm phép toán trên Y
z = y * y * 3
out = z.mean()
# print(z, out) 

a = torch.randn(2,2)
a = ((a * 3) / (a - 1))
a.requires_grad_(True)
b = (a * a).sum()
# print(b.grad_fn)

# Độ dốc (grandient)
# -- Ta tiến hành lan truyền ngược (backprop) vì out chứa 1 giá trị scalar, out.backward() <=> out.backward(torch.tensor(1.))
out.backward()
# print(x.grad)

# VD tích vecto Jacobi
d = torch.randn(3, requires_grad=True)
e = d * 2
while e.data.norm() < 1000:
    e = e * 2
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
e.backward(v)
e = d.detach()
print(e.requires_grad)
print(d.eq(e).all())