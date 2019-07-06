# coding=utf-8

import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x_data, y_data = [], []
# 随机生成30个点
for x in range(-10, 20):
    y = -  x ** 2 + 5 * x - 10 + random.random() * 20
    x_data.append([x])
    y_data.append([y])

# 特征构造
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x_data)
# 创建线性模型
linear_reg = LinearRegression()
linear_reg.fit(x_poly, y_data)
plt.plot(x_data, y_data, 'b.')
# 用特征构造数据进行预测
plt.plot(x_data, linear_reg.predict(poly_reg.fit_transform(x_data)), 'r')
plt.show()

