import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from scipy.ndimage import zoom
import numpy.ma as ma
import seaborn as sns


mask = np.load('../Mask/mask.npy')
#
#加载绘制的数据
# raw_matrix_64 = np.load('new_R2_0.95&RMSE_0.65.matrix.npy')
raw_matrix_64 = np.load('new_R2_0.95&RMSE_0.65.matrix.npy')


for i in range(64):
    for j in range(64):
        if mask[i, j] == True:
            raw_matrix_64[i, j] = 1


# 计算平均温度数据
raw_data = np.load('../模型结果/DatLSTM/20240701/reverse_pred_DatLSTM_BoHai_sst.npy')
# print(raw_data.shape)
mean_sst1 = np.zeros([64, 64])

# 最大值最小值
vmax = 10
vmin = 0
# 设置颜色分层数目
sst_levels = np.linspace(vmin, vmax, 10)

for i in range(64):
    for j in range(64):
        # 平均温度单位从k转换成摄氏度
        mean_sst1[i, j] = raw_data[:,:, i, j].mean()
# 利用掩码遮盖
mean_sst = np.where(mask == True, np.nan, raw_matrix_64)
# 创建图形和子图
fig, axs = plt.subplots(1, 1, figsize=(8, 8))
# 温度分布图

figData = ma.masked_array(raw_matrix_64, mask)
# figData = raw_matrix_128

# 带掩码数组
im = axs.contourf(figData, cmap='coolwarm', levels=sst_levels, vmax=vmax, vmin=vmin)

# # 绘制颜色条
cbar = plt.colorbar(im, ax=axs, fraction=0.050, pad=0.03, ticks=np.arange(vmin+1, vmax, 1), format='%d'+' Day')

# 设置颜色条刻度标签的字体大小
cbar.ax.tick_params(labelsize=12)


# 设置左边子图的刻度
# xticks = [0, 20, 40, 60, 80, 100, 120]  # 自定义X轴刻度位置
# yticks = [123, 103, 83, 63, 43, 23, 3]  # 自定义Y轴刻度位置
# axs.set_xticks(xticks)
# axs.set_yticks(yticks)

# 设置左边子图的刻度标签
# X轴刻度和标签
# xticklabels = [ '118°E', '119°E', '120°E', '121°E','122°N']  # 自定义X轴刻度标签
# yticklabels = ['37.5°N', '38.5°N', '39.5°N','40.5°N', '41.5°N']  # 自定义Y轴刻度标签
# axs.set_xticklabels(xticklabels)
# # axs.set_yticklabels(yticklabels)
xticklabels = ['118.8°E', '119.3°E', '119.8°E', '120.3°E', '120.8°E','121.3°E','121.8°E']  #,'122.3°E'
axs.set_xticklabels(xticklabels)
yticklabels = ['37.5°N', '38.0°N', '38.5°N', '39.0°N', '39.5°N', '40.0°N', '40.5°N']
axs.set_yticklabels(yticklabels)
# 设置x轴和y轴刻度的字体大小
axs.tick_params(axis='x', labelsize=12)  # 设置x轴刻度字体大小为12
axs.tick_params(axis='y', labelsize=12)  # 设置y轴刻度字体大小为12

# 绘制25摄氏度的等值线
# levels = [25.5]
# contour = axs.contour(mean_sst1, levels, colors='black', linestyles='solid', linewidths=2.5)

# 添加字体
plt.text(40, 20, '1', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(46, 18, '1', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
# plt.text(48, 101, '1', horizontalalignment='center', verticalalignment='bottom', fontsize=15)
#
plt.text(36, 20, '2', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(56, 23, '2', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(44, 30, '2', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(53, 15, '2', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
# #
plt.text(58, 20, '3', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(52, 23, '3', horizontalalignment='center', verticalalignment='bottom', fontsize=12)

# plt.text(58, 70, '3', horizontalalignment='center', verticalalignment='bottom', fontsize=15)
# #
plt.text(14, 25, '4', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(22, 25, '4', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
# #
plt.text(20, 30, '5', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(4, 29, '5', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(50, 48.5, '5', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(29, 26, '5', horizontalalignment='center', verticalalignment='bottom', fontsize=12)

# #
plt.text(11, 21.5, '6', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(19, 19, '6', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(29, 10, '6', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(16, 30, '6', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(40, 43, '6', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(32.5, 40.5, '6', horizontalalignment='center', verticalalignment='bottom', fontsize=12)


# #
plt.text(24, 35, '7', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(34, 43.5, '7', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(28, 13, '7', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(34.5, 12.5, '7', horizontalalignment='center', verticalalignment='bottom', fontsize=12)

#
plt.text(28, 36.5, '8', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(53, 44, '8', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(48, 53.5, '8', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(36, 29.8, '8', horizontalalignment='center', verticalalignment='bottom', fontsize=12)


plt.text(29.5, 42, '9', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(35, 31.5, '9', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(48.5, 38.5, '9', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(40.8, 47, '9', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
plt.text(47.8, 51.5, '9', horizontalalignment='center', verticalalignment='bottom', fontsize=12)


# 调整子图间的距离
plt.subplots_adjust(wspace=0.1)

plt.savefig('../数据分析结果/20240701/' + 'Fig8.pdf', dpi=300, bbox_inches='tight',format='pdf')
plt.savefig('../数据分析结果/20240701/' + 'Fig8.png', dpi=300, bbox_inches='tight', format='png')

# 显示图形
plt.show()





