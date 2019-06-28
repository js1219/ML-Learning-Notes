# coding=utf-8

import random
import matplotlib.pyplot as plt

# 随机生成30个点
x_data, y_data = [], []
for x in range(0, 30):
    y = x * 0.2 + 2 + random.random()
    x_data.append(x)
    y_data.append(y)

# 学习率
lr = 0.001
# 斜率
k = 0
# 截距
b = 0
# 迭代次数
epocsh = 10000


# 定义损失函数
def cost():
    total_cost = 0
    for i in range(0, len(x_data)):
        total_cost += (k * x_data[i] + b - y_data[i]) ** 2
    return total_cost / len(x_data)


# 开始迭代学习
for i in range(0, epocsh):
    b_grad = 0
    k_grad = 0
    for j in range(0, len(x_data)):
        # 损失函数对b求偏导数
        b_grad += -(y_data[j] - (k * x_data[j] + b))
        # 损失函数对k求偏导数
        k_grad += -x_data[j] * (y_data[j] - (k * x_data[j] + b))
    # 学习率*偏导数的平均值
    b += -lr * (b_grad / len(x_data))
    k += -lr * (k_grad / len(x_data))
    # 每迭代1000次计算一次损失函数
    if i % 1000 == 0:
        print(cost())

# 画原始点
plt.plot(x_data, y_data, 'b.')
# 画拟合的直线
plt.plot(x_data, [k * x + b for x in x_data], 'r')
plt.show()
