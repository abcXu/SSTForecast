import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#数据顺序['ConvLSTM', 'PredRNN', 'MoDeRNN', 'ACFN']
mae_list = np.load('../数据分析结果/20240701/days_r_2_later.npy')
max_value = np.max(mae_list)
min_value = np.min(mae_list)

print(f"最大值: {max_value}")
print(f"最小值: {min_value}")
# 定义三个列表，用于存放不同算法的10天误差图
mask = np.load('D:/Study/SwinLSTM-main/Mask/mask.npy')

# colors = ['#FF3B00','#FFB200', '#FEED00', '#CAFF2C', '#30FFC7', '#00E1FB', '#00A1FF', '#0021FF']

colors = ['#FF0000','#FF5B00', '#FF8800', '#FFAA00', '#FFCC00', '#FEED00', '#CAFF2C',
          '#97FF60',  '#49FFAD', '#30FFC7', '#00E1FB', '#00A1FF', '#0061FF', '#0021FF','#1D10CC','#000088']


# colors = ['#DC0000','#FF0000','#FF3B00', '#FF7700', '#FF9400', '#FFB200', '#FEED00', '#CAFF2C', '#97FF60',
#           '#49FFAD', '#30FFC7', '#00E1FB', '#00A1FF', '#0061FF', '#0021FF','#1D10CC']

# colors = ['#800000',  '#FF0000','#DC143C','#CD5C5C','#F08080','#FA8072','#FF8C00','#FFA500','#FFD700','#FFFF00',
#           '#FFFFE0','#FFFACD','#90EE90','#008000','#008080','#40E0D0',
#           '#87CEEB','#00BFFF','#0000FF','#00008B']
# colors = ['#FF00FF', '#8000FF', '#0021FF', '#00A1FF', '#00E1FB', '#30FFC7', '#CAFF2C', '#FEED00', '#FFB200', '#FF3B00']#0021FF'3194E71D10CC
colors = colors[::-1]
# 创建包含四个子图的图像
fig, axes = plt.subplots(nrows=5, ncols=10,  figsize=(16, 8),dpi=300)
# 定义vmax vmin
vmax = 1
vmin = 0.75
order = ['st', 'nd', 'rd', 'th','th', 'th','th', 'th','th', 'th']
# 绘制convlstm
for i in range(10):

    sns.heatmap(np.flipud(mae_list[i, 0,  ...]), ax=axes[0][i], mask = np.flipud(mask), cbar=False, cmap=colors,
                xticklabels=False, yticklabels=False, square=True, vmax = vmax, vmin = vmin)
    axes[0][i].set_title('$'+str(i+1)+'^{'+order[i]+'}$ day', y=1.0, fontdict={'family':'Times New Roman', 'size':15, 'weight': 'bold'})

    axes[0][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[0][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[0][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[0][i].spines['right'].set_visible(True)  # 显示底部轴线


    sns.heatmap(np.flipud(mae_list[i, 1,  ...]), ax=axes[1][i], mask = np.flipud(mask) ,cbar=False, cmap=colors,
                xticklabels=False, yticklabels=False, square=True, vmax = vmax, vmin = vmin)

    axes[1][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[1][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[1][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[1][i].spines['right'].set_visible(True)  # 显示底部轴线
    #

    sns.heatmap(np.flipud(mae_list[i, 2,  ...]), ax=axes[2][i], mask = np.flipud(mask) ,cbar=False, cmap=colors, xticklabels=False, yticklabels=False, square=True, vmax = vmax, vmin = vmin)
    # 设置边框
    axes[2][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[2][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[2][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[2][i].spines['right'].set_visible(True)  # 显示底部轴线

    sns.heatmap(np.flipud(mae_list[i, 3,  ...]), ax=axes[3][i], mask = np.flipud(mask) ,cbar=False, cmap=colors, xticklabels=False, yticklabels=False, square=True, vmax = vmax, vmin = vmin)
    # 设置边框
    axes[3][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[3][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[3][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[3][i].spines['right'].set_visible(True)  # 显示底部轴线

    sns.heatmap(np.flipud(mae_list[i, 4, ...]), ax=axes[4][i], mask=np.flipud(mask), cbar=False, cmap=colors,
                xticklabels=False, yticklabels=False, square=True, vmax=vmax, vmin=vmin)
    # 设置边框
    axes[4][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[4][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[4][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[4][i].spines['right'].set_visible(True)  # 显示底部轴线
    #
    # sns.heatmap(np.flipud(mae_list[i, 5, ...]), ax=axes[5][i], mask=np.flipud(mask), cbar=False, cmap=colors,
    #             xticklabels=False, yticklabels=False, square=True, vmax=vmax, vmin=vmin)
    # # 设置边框
    # axes[5][i].spines['left'].set_visible(True)  # 显示左边轴线
    # axes[5][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    # axes[5][i].spines['top'].set_visible(True)  # 显示顶部轴线
    # axes[5][i].spines['right'].set_visible(True)  # 显示底部轴线
    #
    # sns.heatmap(np.flipud(mae_list[i, 3, ...]), ax=axes[6][i], mask=np.flipud(mask), cbar=False, cmap=colors,
    #             xticklabels=False, yticklabels=False, square=True, vmax=vmax, vmin=vmin)
    # # 设置边框
    # axes[6][i].spines['left'].set_visible(True)  # 显示左边轴线
    # axes[6][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    # axes[6][i].spines['top'].set_visible(True)  # 显示顶部轴线
    # axes[6][i].spines['right'].set_visible(True)  # 显示底部轴线

# 在每行的第一个子图中添加文本标签['ConvLSTM','SwinLSTM','ACFN','SE_ConvLSTM','SK_ConvLSTM']
axes[0][0].text(-0.8, 0.5, 'ConvLSTM', size=17, ha='center', va='center', transform=axes[0][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'size':17, 'weight': 'bold'})
axes[1][0].text(-0.8, 0.5, 'PredRNN', size=17, ha='center', va='center', transform=axes[1][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'size': 17, 'weight': 'bold'})
axes[2][0].text(-0.8, 0.5, 'TCTN', size=17, ha='center', va='center', transform=axes[2][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'size': 17, 'weight': 'bold'})
axes[3][0].text(-0.8, 0.5, 'SwinLSTM', size=17, ha='center', va='center', transform=axes[3][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'size': 17, 'weight': 'bold'})
axes[4][0].text(-0.8, 0.5, 'DatLSTM', size=17, ha='center', va='center', transform=axes[4][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'size': 17, 'weight': 'bold'})
# axes[5][0].text(-0.4, 0.5, 'SimVP', fontsize=22, ha='center', va='center', transform=axes[5][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'size': 6, 'weight': 'bold'})
# axes[6][0].text(-0.4, 0.5, 'ACFN', fontsize=22, ha='center', va='center', transform=axes[6][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'size': 6, 'weight': 'bold'})
# 调整子图之间的距离
# plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.subplots_adjust(wspace=0.1, hspace=0.2)






# cbar = fig.colorbar(axes[0,0].imshow([[vmin, vmax]],
#                    cmap=plt.get_cmap('rainbow')), cax=cbar_ax)
cbaxes = fig.add_axes([0.92, 0.2, 0.02, 0.6])
cbar = plt.colorbar(axes[0,0].imshow([[vmin, vmax]],
                  cmap=plt.get_cmap('rainbow')),cax=cbaxes,
                     orientation= 'vertical')
cbar.ax.set_position([0.92, 0.1, 0.02, 0.8])
# 设置颜色条刻度和标签
# cbar_ticks = [0.30℃, 0.66℃, 1.02, 0.8,1.0,1.2, 1.4,1.6, 1.8, 2.0,2.2, 2.4, 2.6,2.8, 3.0]
cbar_ticks = np.linspace(0.75, 1, 9).tolist()
char_ticks=[]

# cbar.set_ticks([(x - vmin) / (vmax - vmin) for x in cbar_ticks])
# cbar.set_ticklabels([f"{x:.1f}" for x in cbar_ticks])
cbar.set_ticks(cbar_ticks)
# cbar.ax.tick_params(labelsize=8)
colorbar = fig.colorbar(axes[0][0].collections[0], ax=axes.ravel().tolist(), cax=cbaxes, shrink=0.90, aspect=10,ticks=cbar_ticks)
cbar.set_ticklabels([f"{x:.2f}" for x in cbar_ticks])
# colorbar.ax.tick_params(labelsize=20)  # 设置图例刻度标签的字体大小为12
# colorbar.ax.yaxis.label.set_weight('bold')  # 设置标签加粗
# # 添加共享的颜色条℃
# cbar_ticks = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
# # cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
# cbar_ax = fig.add_axes([0.915, 0.12, 0.015, 0.75])
# colorbar = fig.colorbar(axes[0][0].collections[0], ax=axes.ravel().tolist(), cax=cbar_ax, shrink=0.90, aspect=10)
colorbar.ax.tick_params(labelsize=18)  # 设置图例刻度标签的字体大小为12
colorbar.ax.yaxis.label.set_weight('bold')  # 设置标签加粗









# # 添加共享的颜色条
# cbar_ax = fig.add_axes([0.915, 0.12, 0.015, 0.75])
# colorbar = fig.colorbar(axes[0][0].collections[0], ax=axes.ravel().tolist(), cax=cbar_ax, shrink=0.90, aspect=10)
# colorbar.ax.tick_params(labelsize=20)  # 设置图例刻度标签的字体大小为12
# colorbar.ax.yaxis.label.set_weight('bold')  # 置标签加粗

# fig.colorbar(axes[1][0].collections[0], ax=axes.ravel().tolist(), shrink=0.90, aspect=10)

# # # plt.savefig('图片/Fig3-MAE.eps', format='eps', dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig('D:/Study/SwinLSTM-main/数据分析结果/20240701/Fig55.eps', format='eps',dpi=300, bbox_inches='tight', pad_inches=0)
# plt.savefig('D:/Study/SwinLSTM-main/数据分析结果/20240701/Fig55.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0)
# plt.savefig('D:/Study/SwinLSTM-main/数据分析结果/20240701/Fig55.pdf', format='png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig('D:/Study/SwinLSTM-main/数据分析结果/20240701/Fig55-eps-converted-to.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)

# # plt.savefig('MAE.png', format='png', transparent=True, dpi=600, bbox_inches='tight', pad_inches=0)
# plt.savefig('MAE.png', format='png',  dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()
