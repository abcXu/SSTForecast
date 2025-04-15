import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载原始数据和预测数据
original_data = np.load('../模型结果/DatLSTM/20240701/reverse_true_DatLSTM_BoHai_sst.npy')
predicted_data = np.load('../模型结果/DatLSTM/20240701/reverse_pred_DatLSTM_BoHai_sst.npy')

# 计算有效预测时长
effective_duration = np.abs(original_data - predicted_data)
effective_duration = np.mean(effective_duration, axis=1)  # 取平均值,得到(2919, 64, 64)的数组
effective_duration = np.mean(effective_duration, axis=0)  # 再次取平均值,得到(2919,)的数组

# 创建网格坐标
x, y = np.meshgrid(np.arange(64), np.arange(64))

# 扩展 effective_duration 的形状
effective_duration = np.expand_dims(effective_duration, axis=(1, 2))

# 绘制平面分布图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, effective_duration, cmap='viridis')

# 设置标签和标题
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('有效预测时长')
ax.set_title('海洋温度有效预测时长平面分布图')

# 显示图像
plt.show()