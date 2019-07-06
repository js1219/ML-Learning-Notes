# coding=utf-8

import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x_data, y_data = [], []
# 随机生成30个点
for x in range(1, 30):
    y = x * 2 + 5 + (float('%.2f' % random.random()) * 10 - 5)
    x_data.append([x])
    y_data.append([y])

# 创建线性模型
mode = LinearRegression()
# 训练
mode.fit(x_data, y_data)
# 画图
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, mode.predict(x_data), 'r')
plt.show()
