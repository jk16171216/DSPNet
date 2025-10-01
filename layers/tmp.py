import numpy as np
import matplotlib.pyplot as plt

# 数据准备
categories = ['ETT', 'PEMS', 'Solar', 'Weather', 'Traffic', 'ECL']
data = {
    'iTransformer': [0.38, 0.12, 0.23, 0.26, 0.43, 0.18],
    'PatchTST': [0.38, 0.22, 0.28, 0.26, 0.55, 0.23],
    'DLinear': [0.38, 0.32, 0.33, 0.46, 0.67, 0.23],
    'Transformer': [0.38, 0.22, 0.28, 0.33, 0.66, 0.23],
    'TimesNet': [0.38, 0.12, 0.23, 0.26, 0.66, 0.18],
    'FEDformer': [0.38, 0.22, 0.28, 0.26, 0.43, 0.23]
}

N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# 绘图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw = {'polar': True})

for key, value in data.items():
    values = value + value[:1]
    ax.plot(angles, values, label=key)
    ax.fill(angles, values, alpha=0.2)

ax.set_thetagrids([angle * 180 / np.pi for angle in angles[:-1]], categories)
ax.set_title('Radar Chart Example', pad=20)
ax.legend()
plt.show()