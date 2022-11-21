# Mạng nở ron 
# -- Có thể xây dựng thông qua gói torch.nn
# -- torch.nn dựa trên autograd để đ/n mô hình và tính đạo hàm

# Quá trình huấn luyện mạng nơ ron như sau:
# -- Đ/n 1 mạng nơ ron với các tham số có thể học trong quá trình huấn luyện
# -- Lặp qua 1 dataset chứa các input
# -- Xử lý input qua mạng nơ ron
# -- Tính giá trị loss (output sai khác bn so vs gt đúng)
# -- Lan truyền ngược độ dóc để cập nhập các trọng số 
# -- Cập nhập trọng số của mạng với quy tắc đơn giản: weight = weight - learning_rate * gradient

# Định nghĩa mạng nơ ron

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 1 input image channel , 6 output channel , 3x3 square
        # kernel
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        # y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3= nn.Linear(84,10)
    def forward(self,x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 

net = Net()
input = torch.randn(1,1,32,32)
out = net(input)
net.zero_grad()
out.backward(torch.randn(1,10))
# print(out)

#. Toàn bộ gói torch.nn chỉ hỗ trợ dưới dạng mini-batch chứ ko hỗ trợ một mẫu riêng lẻ
# vd: nn.Conv2d sẽ nhận vào Tensor 4D với các chiều là nSamples x nChannels x Height x Width
#. Nếu ra chỉ có 1 mau thi dùng input.unsqueeze(0) để tạo ra 1 chiều giả

# ÔN TẬP
#-- torch.Tensor - Một mảng đa chiều hỗ trợ phép toán autograd như backward()
#-- nn.Module - Cấu phần giúp tạo mạng nơ ron Một cách đơn giản để gói các trọng số, hỗ trợ đưa lên GPU và các hoạt động liên quan
#-- nn.Parameter - Một loại tensor, được coi là tham số khi trở thành một thuộc tính của một Module
#-- autograd.Function - Đ/n pt lan truyền và lan truyền ngược 1 cách tự động. Mọi phương thức trên tensor sẽ tạo ra 1 nút Function kết nối Tensor vs hàm tạo ra nó và lưu trữ lịch sử tính toán

# Hàm Loss
#-- Nhận đầu vào là cặp (output,target) và tính khoảng cách giữa 2 là bao xa
#-- Hàm nn.MSELoss sẽ tính sai số toàn phương tb giữa chúng

output = net(input)
target = torch.randn(10)
target = target.view(1,-1)
criterion = nn.MSELoss()
loss = criterion(output,target)
# print(loss)

#. => biểu đồ tính toán: input->conv2d->relu->maxpol2d->conv2d->relu->maxpool2d->view->linear->relu->linear->relu->linear->MSELoss->loss
#-- khi ta gọi loss.backward toàn bộ biểu đồ đc tính đạo hàm,

# print(loss.grad_fn)
# print(loss.grad_fn.next_functions[0][0])
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

# Lan Truyền Ngược
#- Để lan truyền ngược lỗi => loss.backward() và cần xóa các giá trị gradient đã có để tránh cộng dồn

net.zero_grad() # xỏa tât cả các gt grandient
# print(net.conv1.bias.grad)
loss.backward()
# print(net.conv1.bias.grad)

# Cập nhập trọng số

# w = w - learning_rate * grandient
optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
output = net(input)
loss = criterion(output,target)
loss.backward()
optimizer.step()