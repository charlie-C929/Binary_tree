import numpy as np
import math
import matplotlib.pyplot as plt

e = math.e


def tanh(x):
    return (e ** x - e ** (-x)) / (e ** x + e ** (-x))


def softplus(x):
    return math.log(1 + pow(e, x))


def mish(x):
    return x * tanh(softplus(x))


x = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 5, 1000)
for i in range(1000):
    y[i] = mish(x[i])

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.set_xticks([-10, -5, 0, 5, 10])
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
ax.set_yticks([-10, -5, 5, 10])

plt.plot(x, y, color='red', linewidth=3, label='Mish')
plt.legend()
plt.savefig('mish.jpg')
plt.show()
