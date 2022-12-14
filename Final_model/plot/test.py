import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return (0.3*np.exp(-(x-0.3)**2) + 0.7* np.exp(-(x-2.)**2/0.3))/1.2113
x = np.arange(-4.,6.,0.01)
plt.plot(x,f1(x),color = "red")

size = int(1e+07)
sigma = 1.2
z = np.random.normal(loc = 1.4,scale = sigma, size = size)
qz = 3/(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*(z-1.4)**2/sigma**2)
k = 2.5
#z = np.random.uniform(low = -4, high = 6, size = size)
#qz = 0.1
#k = 10
u = np.random.uniform(low = 0, high = k*qz, size = size)

pz = 0.3*np.exp(-(z-0.3)**2) + 0.7* np.exp(-(z-2.)**2/0.3)
sample = z[pz >= u]
plt.hist(sample,bins=50, normed=True, edgecolor='black')

#设置坐标轴范围
plt.xlim([0.3,4])

#隐藏坐标刻度
plt.xticks([])
plt.yticks([])

plt.show()

a = [1,2,3]

for item in a:
    b=[]
    b.append(item)
    print(item)

print(b)

with open("1.txt") as f:
    f.write('1')

f.close()
