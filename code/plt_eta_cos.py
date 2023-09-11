import pandas as pd

df = pd.read_excel('/Users/boa/Documents/2023-2024-1暑假/CUMCM2023/data/附件.xlsx')

eta_cos_df = pd.read_csv('/Users/boa/Documents/2023-2024-1暑假/CUMCM2023/data/2023-03-21 09:00:00.csv')

# 导入matplotlib.pyplot模块，用于绘图
import matplotlib.pyplot as plt

# 从df中提取x坐标和y坐标的数据
x = df['x坐标 (m)']
y = df['y坐标 (m)']

# 从eta_cos_df中提取eta_cos的数据
eta_cos = eta_cos_df['0.6931308200035265']

# 使用scatter函数绘制散点图，根据eta_cos的值设置颜色和透明度
plt.scatter(x, y, c=eta_cos, alpha=0.8)

# 设置标题和坐标轴标签
plt.title('eta_cos scatter plot')
plt.xlabel('x (m)')
plt.ylabel('y (m)')

# 使用colorbar函数添加颜色条，显示颜色与数值的关系
plt.colorbar(label='eta_cos')

# 显示图像
plt.show()

