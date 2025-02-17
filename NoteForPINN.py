x = 3 # int
y = 6.2
y2 = '6.3'
y4= "99999.43"
z5 = '433.6654'
yy = float(y2)+float(y4)+float(z5)
print(yy)
z1 = 'S'
z2 = "s" # 
t = True
n = [2, 5, 9, 65]
mmm = 7.8356778
print('z1:', z1, "n : ", n)
print(f'z1 : {z1} n : {n}')
print(f'mmm : {mmm:.2f}')
print(z2)
'''
Ali and Ehsan
are
. 
.
.
'''
"""
Ali va Ehsan
are
. 
.
.
"""
print(type(z1))
print(type(z2))
print(n[-1])
m = {'Ali':[20, 19.5, 18],'Ehsan':19}
print(type(m))
print(f'm : {m["Ali"]}')

# ss = int(input('Enter Password :'))

s2 = 0
for i in range(10):
    # s2 = s2 + i
    s2 += i
    if s2 == 21:
       print('Vay') 
    else:
        print('sssss')
    print(f's2 in for : {s2}')
print(f's2 out for : {s2}')

for i in [16, 'Ali', 50]:
    if i == 16:
        print(f'i : {i}')
    elif i == 'Ali':
        print('Salam')
    else:
        print('Hi')


s3=0
while 2>1:
    s3+=1
    if s3 == 10:
        print(f's3 : {s3}')
        break
    

s = 0
for i in range(5):
    s += i
    print(f's = {s}')
print(f's = {s}')

s = []
for i in range(5):
    s.append(i)
    print(f's = {s}')
print(f's = {s}')

ss = [_ for _ in range(5)]
print(f'ss = {ss}')

sss = 0
i = 0
while i <= 4:
    sss += i**2
    i += 1
    print(f'sss = {sss}')
print(f'sss = {sss}')

def add(x,y):
    return x + y**2
print(f'add = {add(3,5)}')

add = lambda x,y: x+y**3
print(f'add = {add(3,2)}')

class HeatTransfer:
    def __init__(self, Tem, mass, capacity):
        self.Tem = Tem
        self.mass = mass
        self.capacity = capacity

    def calcul(self, Tem2):
        Tem1 = self.Tem
        HeatRate = self.mass * self.capacity * (Tem2 - Tem1)
        return HeatRate
    
# if __name__ == "__main__":
Heat = HeatTransfer(Tem=300, mass = 2, capacity = 3)
tem2 = 310
Q = Heat.calcul(tem2)
print(f'Q = {Q}')

import numpy as np

x = np.sin(2*np.pi)

print(x)

import time

# Create a 3D NumPy array and a 3D list
size = 100
numpy_array = np.random.rand(size, size, size)  # 100x100x100 array
list_3d = [[[np.random.rand() for _ in range(size)] for _ in range(size)] for _ in range(size)]

# NumPy array addition
start_time = time.time()
result_numpy = numpy_array + numpy_array
numpy_time = time.time() - start_time

# 3D list addition
start_time = time.time()
result_list = [[[list_3d[i][j][k] + list_3d[i][j][k] for k in range(size)] for j in range(size)] for i in range(size)]
list_time = time.time() - start_time

print(f"NumPy time: {numpy_time:.6f} seconds")
print(f"3D list time: {list_time:.6f} seconds")

LL = [2, 8]
LN1 = np.array(LL)
LN2 = np.array([3, 76, 'uu'])
print(f'ln2 : {LN2}')
lop = LN1*2
print(lop)

bb = np.arange(1,7,0.1) #  np.arange(start, stop, step)
print(bb)

cc=np.linspace(1,7,2) #  np.linspace(start, stop, num)
print(cc)

arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f'arr2d[1, 2], arr2d[0:2, 1:3] : {arr2d[1, 2]} , {arr2d[0:2, 1:3]}')
mm = np.array([[[2, 6],[3,8],[4,8]]])
mn = np.array([[2, 6, 9],[3,8,5]])
print(mn)
print(f'mm.size : {mm.size}')  # .shape # .ndim
print(f'mn.ndim : {mn.ndim}')

vv = np.ones((3,5,4))*300 # np.zeros()
print(vv) # 3 Sets, 5 Rows per Set, 4 Columns

vb = np.full((3,2), 33)
print(vb)

np.random.seed(57776)
# vv = np.random.randint(1,10,size=(2,3))
vv = np.random.randn(1,10)
print(vv)

x1= np.array([[1,2,3]])
x2= np.array([[4,5,6]])
x3=np.concatenate((x1,x2),axis=1)
print(x3)

x1= np.array([[1,2,3]])
np.savetxt('D:\Ehsan.txt',x1)
ss = np.loadtxt('Ehsan.txt')
print(ss)

# np.where(cond, x, y) x if condition else y
aff = np.array([3, 65, 33, 4])
print(np.where(aff<=30,aff+2,aff*2))

import matplotlib.pyplot as plt

x1= np.array([1,2,3])
x2= np.array([4,5,6])

# plt.scatter(x1,x2)
plt.plot(x1,x2,label='Sh')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
# plt.xlim(xmin = 0)
# plt.ylim(ymin = 4,ymax=6)
plt.grid()
plt.show()

import pandas as pd

d = {'Ali': [24, 211, 32], 'Ehsan': 29}   
print(d['Ali'])

data = pd.read_csv('E:\C_battery.csv')
# data = pd.read_excel('/content/C_battery.xlsx', sheet_name='C_battery')
capacity = data['capacity']  #data['voltage mean'].values
print(len(capacity))
print(data['capacity'].max()) 

cycle = np.linspace(0, 393, 393)
print(len(cycle))

plt.plot(cycle, capacity)
plt.xlabel('Cycle')
plt.ylabel('Capacity')
# plt.ylim([1.55,2.05])
plt.grid()
plt.savefig("E:\Capaci.png", dpi=600)
plt.show()


voltage = data['voltage mean']

max_capacity_index = data['capacity'].idxmax()
print(max_capacity_index)
voltage_at_max_capacity = data.loc[max_capacity_index, 'voltage mean']
print(f"Voltage at maximum capacity: {voltage_at_max_capacity:.3f}")

plt.scatter(capacity, voltage)
plt.xlabel('capacity')
plt.ylabel('voltage mean')
plt.show()


f = data.iloc[:,:-1]
f_norm = 2*(f - f.min())/(f.max() - f.min()) - 1
# f_norm = (f_df - f_df.mean())/f_df.std()
print(f'f_norm : {f_norm}')
print(f'f_norm.shape : {f_norm.shape}')

print(data.iloc[:,-1].values)

x = data.iloc[:,-1].values

import torch
tensor_x = torch.from_numpy(x)   # .view(-1,1)
print(tensor_x)

df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
df.to_csv('output.csv') #, index=False

