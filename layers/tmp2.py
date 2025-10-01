import numpy as np
import matplotlib.pyplot as plt
import pickle
from statsmodels.tsa.seasonal import seasonal_decompose

# 生成随机折线的点数
num_points = 48  # 原始点数
x = np.linspace(0, 10, num_points)  # 生成等距递增的 x 轴
y = np.random.rand(num_points) * 10  # y 轴随机值

# 保存 y 值
with open('y_values.pkl', 'wb') as f:
    pickle.dump(y, f)

# 提取 y 值
with open('y_values.pkl', 'rb') as f:
    y = pickle.load(f)

# DownSampling 操作（取一部分点）
downsample_rate = 1
x_downsampled = x[::downsample_rate]
y_downsampled = y[::downsample_rate]

# 进行趋势和季节性分解
result = seasonal_decompose(y, model='additive', period=6)  # 假设周期为5

# 绘制趋势图
plt.figure(figsize=(8, 2))
plt.plot(result.trend, marker='', linestyle='-', color='r', label='Trend', lw=2)
plt.xticks([])
plt.yticks([])
plt.box(False)
plt.show()

# 绘制季节性图
plt.figure(figsize=(8, 2))
plt.plot(result.seasonal, marker='', linestyle='-', color='b', label='Seasonality', lw=2)
plt.xticks([])
plt.yticks([])
plt.box(False)
plt.show()

# 绘制随机折线
plt.figure(figsize=(8, 2))
plt.plot(x_downsampled, y_downsampled, marker='', linestyle='-', color='black', label='Downsampled Zigzag Line', lw=2)
plt.xticks([])  # 移除 x 轴刻度
plt.yticks([])  # 移除 y 轴刻度
plt.box(False)  # 移除边框
plt.show()

# DownSampling 操作（取一部分点）
downsample_rate = 2  # 每隔2个点取一个
x_downsampled = x[::downsample_rate]
y_downsampled = y[::downsample_rate]

# 进行趋势和季节性分解
result = seasonal_decompose(y_downsampled, model='additive', period=6)  # 假设周期为5

# 绘制趋势图
plt.figure(figsize=(4, 2))
plt.plot(result.trend, marker='', linestyle='-', color='r', label='Trend', lw=2)
plt.xticks([])
plt.yticks([])
plt.box(False)
plt.show()

# 绘制季节性图
plt.figure(figsize=(4, 2))
plt.plot(result.seasonal, marker='', linestyle='-', color='b', label='Seasonality', lw=2)
plt.xticks([])
plt.yticks([])
plt.box(False)
plt.show()


# 绘制随机折线
plt.figure(figsize=(4, 2))
plt.plot(x_downsampled, y_downsampled, marker='', linestyle='-', color='black', label='Downsampled Zigzag Line', lw=2)
plt.xticks([])  # 移除 x 轴刻度
plt.yticks([])  # 移除 y 轴刻度
plt.box(False)  # 移除边框
plt.show()

# DownSampling 操作（取一部分点）
downsample_rate = 3  # 每隔4个点取一个
x_downsampled = x[::downsample_rate]
y_downsampled = y[::downsample_rate]

# 进行趋势和季节性分解
result = seasonal_decompose(y_downsampled, model='additive', period=6)  # 假设周期为5

# 绘制趋势图
plt.figure(figsize=(2, 2))
plt.plot(result.trend, marker='', linestyle='-', color='r', label='Trend', lw=2)
plt.xticks([])
plt.yticks([])
plt.box(False)
plt.show()

# 绘制季节性图
plt.figure(figsize=(2, 2))
plt.plot(result.seasonal, marker='', linestyle='-', color='b', label='Seasonality', lw=2)
plt.xticks([])
plt.yticks([])
plt.box(False)
plt.show()

# 绘制随机折线
plt.figure(figsize=(2, 2))
plt.plot(x_downsampled, y_downsampled, marker='', linestyle='-', color='black', label='Downsampled Zigzag Line', lw=2)
plt.xticks([])  # 移除 x 轴刻度
plt.yticks([])  # 移除 y 轴刻度
plt.box(False)  # 移除边框
plt.show()
