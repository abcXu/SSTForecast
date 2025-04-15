import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# mae_list = np.load('数据分析/days_names_mae_heat.npy')
mae_list = np.load('../数据分析结果/20240701/days_mae_later.npy')

# 定义三个列表，用于存放不同算法的10天误差图
mask = np.load('D:/Study/SwinLSTM-main/Mask/mask.npy')

# colors = ['#DC0000','#FF0000','#FF3B00', '#FF7700', '#FF9400', '#FFB200', '#FEED00', '#CAFF2C', '#97FF60',  '#49FFAD', '#30FFC7', '#00E1FB', '#00A1FF', '#0061FF', '#0021FF','#1D10CC']
colors = ['#FF0000','#FF5B00', '#FF8800', '#FFAA00', '#FFCC00', '#FEED00', '#CAFF2C',
          '#97FF60',  '#49FFAD', '#30FFC7', '#00E1FB', '#00A1FF', '#0061FF', '#0021FF','#1D10CC','#000088']
# colors = ['#FF0000','#FF3B00','#FF7700', '#FF9400', '#FFB200', '#FEED00', '#CAFF2C', '#97FF60',
# '#63FF94', '#49FFAD', '#30FFC7', '#00E1FB', '#00A1FF', '#0061FF', '#0021FF']
# colors = ['#FF0000','#FF3300','#FF6600','#FF9900','#FFCC00','#FFFF00','#CCFF00','#99FF00','#66FF00','#00FF00',
#           '#00FF66','#00CCFF','#0099FF','#0066FF','#0033FF','#FF0000','#FF3B00','#FF7700', ]

# colors = ['#FF3B00','#FFB200', '#FEED00', '#CAFF2C', '#30FFC7', '#00E1FB', '#00A1FF', '#0021FF']

# colors = ['#FF00FF', '#8000FF', '#0021FF', '#00A1FF', '#00E1FB', '#30FFC7', '#CAFF2C', '#FEED00', '#FFB200']

# colors = ['#FF0000','#FF1010', '#FF2020', '#FF3030', '#FF4040', '#FF5050', '#FF6060', '#FF7070', '#FF8080',
#           '#FF9090', '#FFA0A0', '#FFB0B0', '#FFC0C0', '#FFD0D0', '#FFE0E0', '#FFF0F0', '#E0E0FF', '#D0D0FF','#C0C0FF', '#B0B0FF']
# '''# colors = ['#800000','#BD0202','#FE0002','#FF4103','#FF7D00','#FFBD00','#C0FE41','#81FF81','#41FEBE','#01FFFF','#02BDFE','#018DFB','#0000FD','#0001BD']
# # colors = ['#FF3B00', '#FF7700', '#FF9400', '#FEED00', '#CAFF2C', '#97FF60', '#30FFC7', '#00E1FB', '#00A1FF', '#0061FF', '#0021FF']
#
# # colors = ['#800000',  '#FF0000','#DC143C','#CD5C5C','#F08080','#FA8072','#FF8C00','#FFA500','#FFD700','#FFFF00',
# #           '#FFFFE0','#FFFACD','#90EE90','#008000','#008080','#40E0D0',
# #           '#87CEEB','#00BFFF','#0000FF','#00008B']
# # '#0000CD ','#006400','#B22222','#8B0000'

colors = colors[::-1]
# 创建包含四个子图的图像
fig, axes = plt.subplots(nrows=5, ncols=10,  figsize=(16, 8),dpi=300)
# 定义vmax vmin
vmax = 3.5
vmin = 0.3      #最小
cbar_yticks = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
max_value = np.max(mae_list)
min_value = np.min(mae_list)
print(f"最大值: {max_value}")
print(f"最小值: {min_value}")
order = ['st', 'nd', 'rd', 'th','th', 'th','th', 'th','th', 'th']
# 绘制convlstm
for i in range(10):

    sns.heatmap(np.flipud(mae_list[i, 0,  ...]), ax=axes[0][i], mask = np.flipud(mask), cbar=False, cmap=colors, xticklabels=False, yticklabels=False, square=True, vmax = vmax, vmin = vmin)
    axes[0][i].set_title('$'+str(i+1)+'^{'+order[i]+'}$ day', y=1.0, fontdict={'family':'Times New Roman', 'size':15, 'weight': 'bold'})

    axes[0][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[0][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[0][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[0][i].spines['right'].set_visible(True)  # 显示底部轴线


    sns.heatmap(np.flipud(mae_list[i, 1,  ...]), ax=axes[1][i], mask = np.flipud(mask) ,cbar=False, cmap=colors, xticklabels=False, yticklabels=False, square=True, vmax = vmax, vmin = vmin)
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

    # sns.heatmap(np.flipud(mae_list[i, 5, ...]), ax=axes[4][i], mask=np.flipud(mask), cbar=False, cmap=colors,
    #             xticklabels=False, yticklabels=False, square=True, vmax=vmax, vmin=vmin)
    # # 设置边框
    # axes[5][i].spines['left'].set_visible(True)  # 显示左边轴线
    # axes[5][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    # axes[5][i].spines['top'].set_visible(True)  # 显示顶部轴线
    # axes[5][i].spines['right'].set_visible(True)  # 显示底部轴线
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
# name_list = ['ConvLSTM','PredRNN','TCTN','SwinLSTM','DatLSTM']
# ['ConvLSTM','PredRNN','TCTN','SwinLSTM','DatLSTM']
axes[0][0].text(-0.8, 0.5, 'ConvLSTM', size=17, ha='center', va='center', transform=axes[0][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'size': 17, 'weight': 'bold'})
axes[1][0].text(-0.8, 0.5, 'PredRNN', size=17, ha='center', va='center', transform=axes[1][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'size': 17, 'weight': 'bold'})
axes[2][0].text(-0.8, 0.5, 'TCTN', size=17, ha='center', va='center', transform=axes[2][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'size': 17, 'weight': 'bold'})
axes[3][0].text(-0.8, 0.5, 'SwinLSTM', size=17,ha='center', va='center', transform=axes[3][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'size': 17, 'weight': 'bold'})
axes[4][0].text(-0.8, 0.5, 'DatLSTM', size=17, ha='center', va='center', transform=axes[4][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'size': 17, 'weight': 'bold'})
# axes[5][0].text(-0.4, 0.5, 'SimVP', size=14, ha='center', va='center', transform=axes[5][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'size': 15, 'weight': 'bold'})
# axes[6][0].text(-0.4, 0.5, 'ACFN', fontsize=22, ha='center', va='center', transform=axes[6][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'size': 6, 'weight': 'bold'})
# 调整子图之间的距离
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
cbar_ticks = np.linspace(0.3, 3.5, 9).tolist()
char_ticks=[]

# cbar.set_ticks([(x - vmin) / (vmax - vmin) for x in cbar_ticks])
# cbar.set_ticklabels([f"{x:.1f}" for x in cbar_ticks])
cbar.set_ticks(cbar_ticks)
# cbar.ax.tick_params(labelsize=8)
colorbar = fig.colorbar(axes[0][0].collections[0], ax=axes.ravel().tolist(), cax=cbaxes, shrink=0.90, aspect=10,ticks=cbar_ticks)
cbar.set_ticklabels([f"{x:.1f}" for x in cbar_ticks])
# colorbar.ax.tick_params(labelsize=20)  # 设置图例刻度标签的字体大小为12
# colorbar.ax.yaxis.label.set_weight('bold')  # 设置标签加粗
# # 添加共享的颜色条℃
# cbar_ticks = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
# # cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
# cbar_ax = fig.add_axes([0.915, 0.12, 0.015, 0.75])
# colorbar = fig.colorbar(axes[0][0].collections[0], ax=axes.ravel().tolist(), cax=cbar_ax, shrink=0.90, aspect=10)
colorbar.ax.tick_params(labelsize=18)  # 设置图例刻度标签的字体大小为12
colorbar.ax.yaxis.label.set_weight('bold')  # 设置标签加粗
#


# # plt.show()
# #
plt.savefig('../数据分析结果/20240701/Fig66.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig('../数据分析结果/20240701/Fig66.eps', format='eps',dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig('../数据分析结果/20240701/Fig66-eps-converted-to.pdf', format='eps',dpi=300, bbox_inches='tight', pad_inches=0)


plt.show()

