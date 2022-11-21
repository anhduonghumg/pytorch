import numpy as np

#. tao marng tu ds
x = np.array([[1,2,4],[3,4,3],[4,5,3]]).shape


#. Dung ham tao mang
zero = np.zeros((2,3)) #. tao ma tran 2 cot 3 hang toan phan tu 0
one = np.ones((2,3)) #. tao ma tran 2 cot 3 hang toafn phan tu 1
arange = np.arange(0,10,2) #. tao marng bat dau tu 0, khoang cach cac gt 2, ket thuc truoc 10
linspace = np.linspace(0,1,4) # tao mang gom 4 phan tu cach deu bao gom ca diem ket thuc
random = np.random.normal(5,6,10) # tao mang cac phan tu ngau nhien

#. lay phan tu trong mang
getData = x[-1]

# dinh hinh lai mang
arrayOne = np.arange(6) # tao mang 1 chieu 6pt
arrayTwo = arrayOne.reshape(2,3) # bien doi sang mang 2 chieu 2x3

# lay dong cuoi
getLast = arrayTwo[-1]
# lay phan tu cuoi cua hang cuoi
getItemLast = arrayTwo[-1,-1]


m = np.arange(10)
# lay ra cac phan tu duoi dang danh sach
# cu phap a:b lay ra tat ca cac pt thoa man
# print(m)
# print(m[:3])
# print(m[1:3])
# print(m[-3:])

# lay phan tu theo thu tu nguoc lai
# cu phap a:b:c voi c la so khoang cach cho phep 
# print(m[::-1])
# print(m[::3])

k = np.arange(25).reshape(5,5)
# print(k)
# print(k[:2,:2]) # lay 2 hang dau va 2 cot dau
# print(k[::2,::2])


z = np.random.randint(0,10,6) # tao mang 1 chieu co 6pt tu 0-9

# tao ra 2 mang
arrayLength = 5
x1 = np.arange(arrayLength)
x2 = np.arange(arrayLength) * -1

# trao mang ngau nhien
indices = np.random.permutation(arrayLength)

# tinh toan tren mang
h = np.array([1,2,3,4,5,6])
# print(h + 10)

# Bai tap
  
#1. Tao mang tu 0 - 26 va gan vao bien n
n = np.arange(27)
#2. Dao nguoc thu tu mang
reverse = n[::-1]
#3 Bien doi sang mang 3 chieu
n3 = n.reshape(3,3,3)
#4. Tim vị tri pt co gt = 12
# pt12 = n.mean(12)
# print(pt12)

#5. Hoán vị ngau nhien 0-10
hv = np.random.permutation(20)

#6. Tao 1 mang cac gia tri logic tu dk pt y > 10
hv10 = hv > 10

#. Tạo một mẫu gồm 20 điểm dữ liệu 
mean = 10
stdev = 5
number_value = 20
k = np.random.normal(mean,stdev,number_value)
print(k)





