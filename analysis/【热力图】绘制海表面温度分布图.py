# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import seaborn as sns
#
# os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
#
# names = ['SSTPM']
# true = np.load('D:/Datasets/result/20250415/reversed/reverse_true_sst_SSTPredictor_mlp_SCS.npy')
#
# SSTPM_pred = np.load('D:/Datasets/result/20250415/reversed/reverse_pred_sst_SSTPredictor_mlp_SCS.npy')
#
# # print('true shape:', true.shape)
#
#
# sample_index = 1200
#
# mask = np.load('D:/GraduationThesis/codes/preData/maskLand=0.npy')
# true = true[sample_index, ...]
# order = ['st', 'nd', 'rd', 'th','th', 'th','th', 'th','th', 'th']
# # 创建包含四个子图的图像
# fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(16, 8), dpi=300)
#
# vmax = true.max()
# vmin = true[true != 0].min()
# print(vmin)
# print(vmax)
# # 定义自定义颜色映射列表
# colors = ['#FF0000','#FF3B00','#FF7700', '#FF9400', '#FFB200', '#FEED00', '#CAFF2C', '#97FF60',
# '#63FF94', '#49FFAD', '#30FFC7', '#00E1FB', '#00A1FF', '#0061FF', '#0021FF']
#
# colors = colors[::-1]
# # 绘制真实值子图
# for i in range(10):
#     sns.heatmap(np.flipud(true[i, ...]), ax=axes[0][i], cbar=False, mask = np.flipud(mask),cmap=colors, xticklabels=False, yticklabels=False, square=True, vmin=vmin, vmax=vmax)
#     axes[0][i].set_title('$'+str(i+1)+'^{'+order[i]+'}$ day', y=1.1, fontdict={'family':'Times New Roman', 'size':15, 'weight': 'bold'})
#
#     # 设置边框
#     axes[0][i].spines['left'].set_visible(True)  # 显示左边轴线
#     axes[0][i].spines['bottom'].set_visible(True)  # 显示底部轴线
#     axes[0][i].spines['top'].set_visible(True)  # 显示顶部轴线
#     axes[0][i].spines['right'].set_visible(True)  # 显示底部轴线
#
#
# # 绘制预测值子图
# for i in range(10):
#
#     # # SSTPM
#     sns.heatmap(np.flipud(SSTPM_pred[sample_index, 9-i, ...]), ax=axes[1][i], mask = np.flipud(mask),cbar=False, cmap=colors, xticklabels=False,
#                 yticklabels=False,square=True, vmin=vmin, vmax=vmax)
#     # 设置边框
#     axes[1][i].spines['left'].set_visible(True)  # 显示左边轴线
#     axes[1][i].spines['bottom'].set_visible(True)  # 显示底部轴线
#     axes[1][i].spines['top'].set_visible(True)  # 显示顶部轴线
#     axes[1][i].spines['right'].set_visible(True)  # 显示底部轴线
#
#
# # 在每行的第一个子图中添加文本标签
#
# axes[0][0].text(-0.8, 0.5, 'OSTIA', fontsize=16, ha='center', va='center', transform=axes[0][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'weight': 'bold'})
# axes[1][0].text(-0.95, 0.5, 'SSTPM(Ours)', fontsize=16, ha='center', va='center', transform=axes[1][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'weight': 'bold'})
#
#
#
# # 调整子图之间的距离
# plt.subplots_adjust(wspace=0.1, hspace=0.25)
#
#
# cbaxes = fig.add_axes([0.92, 0.2, 0.02, 0.6])
# cbar = plt.colorbar(axes[0,0].imshow([[vmin, vmax]],
#                   cmap=plt.get_cmap('rainbow')),cax=cbaxes,
#                      orientation= 'vertical')
# cbar.ax.set_position([0.92, 0.1, 0.02, 0.8])
# #
# ax = plt.gca()
#
# # 在图像右侧添加垂直的数字标注
# cbar = plt.colorbar(cax=ax.figure.add_axes([0.92, 0.1, 0.02, 0.8]), orientation='vertical')
# cbar.set_label('Vertical Numerical\nDescription', rotation=270, labelpad=20)
#
# # 设置刻度和标签
# cbar.set_ticks(np.linspace(cbar.norm.vmin, cbar.norm.vmax, 5))
# cbar.set_ticklabels(['{:.1f}'.format(x) for x in np.linspace(cbar.norm.vmin, cbar.norm.vmax, 5)])
#
# # 调整坐标轴范围和标签
# ax.set_xlim(0, 10)
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_title('My Image')
#
# plt.show()


# 添加共享的颜色条
# cbar_ax = fig.add_axes([0.91, 0.12, 0.015, 0.75])
# cbar = fig.colorbar(axes[0][0].collections[0], ax=axes.ravel().tolist(), cax=cbar_ax, shrink=0.80, aspect=10)
# # cbar.set_ticks([4, 8, 12, 16])
# cbar.set_ticklabels(['4', '8', '12', '16'])
# cbar.ax.tick_params(labelsize=12, colors='black')
# cbar.ax.set_ylabel('SST (°C)', rotation=90, va='bottom', fontsize=14)
# cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
# cbar = fig.colorbar(axes[0][0].collections[0], ax=axes.ravel().tolist(), cax=cbar_ax, shrink=1.0, aspect=10)
# cbar_ax = fig.add_axes([0.91, 0.12, 0.015, 0.75])  #原始
# cbar_ax = fig.add_axes([0.91, 0.12, 0.015, 0.75])
# fig.colorbar(axes[0][0].collections[0], ax=axes.ravel().tolist(), cax=cbar_ax, shrink=0.90, aspect=10)
# plt.savefig('../5个算法结果/20240525/5个算法热力图-picture-1200.eps', format='eps',dpi=300, bbox_inches='tight', pad_inches=0)
# # plt.savefig('../5个算法结果/20240525/5个算法热力图-picture-1200-eps-converted-to.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
# plt.savefig('../5个算法结果/20240525/5个算法热力图2400.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0)
# plt.savefig('../数据分析结果/20240701/20130328新.eps', format='eps',dpi=300, bbox_inches='tight', pad_inches=0)
# plt.savefig('../数据分析结果/20240701/20130328新-eps-converted-to.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()

# if __name__ == '__main__':


import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from matplotlib.colors import ListedColormap

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# === 加载数据 ===
names = ['SSTPM']
true = np.load('D:/Datasets/result/20250415/reversed/reverse_true_sst_SSTPredictor_mlp_SCS.npy')
SSTPM_pred = np.load('D:/Datasets/result/20250415/reversed/reverse_pred_sst_SSTPredictor_mlp_SCS.npy')
sample_index = 1000
mask = np.load('D:/GraduationThesis/codes/preData/maskLand=0.npy')  # shape [64, 64]

# === 数据准备 ===
true = true[sample_index, ...]  # [10, 64, 64]

# 自定义颜色映射
colors = ['#FF0000','#FF3B00','#FF7700', '#FF9400', '#FFB200', '#FEED00', '#CAFF2C', '#97FF60',
'#63FF94', '#49FFAD', '#30FFC7', '#00E1FB', '#00A1FF', '#0061FF', '#0021FF'][::-1]
custom_cmap = ListedColormap(colors)

# 色标范围
vmax = true.max()
vmin = true[true != 0].min()

# 创建图像
fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(16, 8), dpi=300)

# 自动生成序数后缀
def ordinal(n):
    suffix = ["th", "st", "nd", "rd"] + ["th"] * 6
    if 10 <= n % 100 <= 20:
        suffix_idx = 0
    else:
        suffix_idx = min(n % 10, 4)
    return f"${n}" + "^{" + f"{suffix[suffix_idx]}" + "}$ day"

# === 绘制 OSTIA 真值 ===
for i in range(10):
    sns.heatmap(np.flipud(true[i, ...]), ax=axes[0][i], cbar=False, mask=np.flipud(mask==0),
                cmap=custom_cmap, xticklabels=False, yticklabels=False,
                square=True, vmin=vmin, vmax=vmax)
    axes[0][i].set_title(ordinal(i+1), y=1.1, fontdict={'family': 'Times New Roman', 'size': 14, 'weight': 'bold'})
    for side in ['left', 'bottom', 'top', 'right']:
        axes[0][i].spines[side].set_visible(True)

# === 绘制 SSTPM 预测值 ===
for i in range(10):
    sns.heatmap(np.flipud(SSTPM_pred[sample_index, 9-i, ...]), ax=axes[1][i], cbar=False, mask=np.flipud(mask==0),
                cmap=custom_cmap, xticklabels=False, yticklabels=False,
                square=True, vmin=vmin, vmax=vmax)
    for side in ['left', 'bottom', 'top', 'right']:
        axes[1][i].spines[side].set_visible(True)

# === 添加左侧标签 ===
axes[0][0].text(-0.8, 0.5, 'OSTIA', fontsize=16, ha='center', va='center',
                transform=axes[0][0].transAxes, rotation=0,
                fontdict={'family': 'Times New Roman', 'weight': 'bold'})

axes[1][0].text(-0.95, 0.5, 'SSTPM (Ours)', fontsize=16, ha='center', va='center',
                transform=axes[1][0].transAxes, rotation=0,
                fontdict={'family': 'Times New Roman', 'weight': 'bold'})

# === 添加颜色条 ===
cbaxes = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cbaxes, orientation='vertical')
cbar.set_label('SST (°C)', rotation=270, labelpad=15, fontsize=12)
cbar.set_ticks(np.linspace(vmin, vmax, 5))
cbar.set_ticklabels([f'{x:.2f}' for x in np.linspace(vmin, vmax, 5)])

# === 调整整体布局 ===
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()


