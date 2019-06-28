# coding=utf-8

import random
import matplotlib.pyplot as plt

# 随机生成 30 个点
x_data, y_data = [], []
for x in range(-10, 20):
    y = x ** 2 - 8 * x - 2 + random.random() * 10
    x_data.append(x)
    y_data.append(y)

# 学习率
lr = 0.00003
# 截距
b = 0
# 斜率
k1 = 0
k2 = 0

# 迭代次数
epocsh = 10000


# 定义损失函数
def cost():
    total_cost = 0
    for i in range(0, len(x_data)):
        x = x_data[i]
        y = y_data[i]
        total_cost += (k1 * x ** 2 + k1 * x + b - y) ** 2
    return total_cost / len(x_data)


for i in range(0, epocsh):
    b_grad = 0
    k1_grad = 0
    k2_grad = 0
    for j in range(0, len(x_data)):
        x = x_data[j]
        y = y_data[j]

        # 损失函数对b求偏导数
        b_grad += k1 * x ** 2 + k2 * x + b - y
        # 损失函数对k1求偏导数
        k1_grad += x ** 2 * (k1 * x ** 2 + k2 * x + b - y)
        # 损失函数对k2求偏导数
        k2_grad += x * (k1 * x ** 2 + k2 * x + b - y)

    # 学习率*偏导数的平均值
    k1 += -lr * k1_grad / len(x_data)
    k2 += -lr * k2_grad / len(x_data)
    b += -lr * b_grad / len(x_data)
    # 每迭代1000次计算一次损失函数
    if i % 1000 == 0:
        print(cost())

# 画原始点
plt.plot(x_data, y_data, 'b.')
# 画拟合的线
plt.plot(x_data, [k1 * x ** 2 + k2 * x + b for x in x_data], 'r')
plt.savefig('1.png')
plt.show()
