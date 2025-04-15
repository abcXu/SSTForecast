import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
#总数据：14134

# names = ['ConvLSTM','MoDeRNN', 'TCTN', 'SwinLSTM']
# names = ['ConvLSTM','SwinLSTM', 'ACFN','SE_ConvLSTM','SK_ConvLSTM']
names = ['ConvLSTM','PredRNN','TCTN','SwinLSTM','DatLSTM']
# ['ConvLSTM','SwinLSTM','ACFN','SE_ConvLSTM','SK_ConvLSTM']
# names = ['ConvLSTM','TCTN','SwinLSTM']

# true = np.load('F:/会议_实验数据/未考虑瑞年情况/experiment_result/ConvLSTM/reverse/masked_data/true_masked_data.npy')
# true = np.load('../5个算法/ConvLSTM/20240525/reverse_true_ConvLSTM_BoHai_sst.npy')
true = np.load('../模型结果/DatLSTM/20240701/reverse_true_DatLSTM_BoHai_sst.npy')
# true = np.load('./data/20240321/reverse-BohaiSea_50epoch_true.npy')

# PredRNN_pred = np.load('F:/实验数据/experiment_result/PredRNN/reverse/masked_data/pred_masked_data.npy')
# ConvLSTM_pred = np.load('F:/实验数据/experiment_result/ConvLSTM/reverse/masked_data/pred_masked_data.npy')
# MoDeRNN_pred = np.load('F:/实验数据/experiment_result/MoDeRNN/reverse/masked_data/pred_masked_data.npy')

# PredRNN_pred = np.load('F:/会议_实验数据/未考虑瑞年情况/experiment_result/PredRNN/reverse/masked_data/pred_masked_data.npy')
# ConvLSTM_pred = np.load('../5个算法/ConvLSTM/20240525/reverse_pred_ConvLSTM_BoHai_sst.npy')
# SwinLSTM_pred = np.load('../5个算法/SwinLSTM/20240525/reverse_pred_SwinLSTM_BoHai_sst.npy')
# ACFN_pred = np.load('../5个算法/ACFN/20240525/reverse_pred_ACFN_Bohai_sst.npy')
# SE_ConvLSTM_pred = np.load('../5个算法/SE_ConvLSTM/20240525/reverse_pred_SE_ConvLSTM_Bohai_sst.npy')
# SK_ConvLSTM_pred = np.load('../5个算法/SK_ConvLSTM/20240525/reverse_pred_SK_ConvLSTM_Bohai_sst.npy')
# SwinLSTM_pred = np.load('./data/20240321/reverse-BohaiSea_50epoch_pred.npy')
# 小论文
ConvLSTM_pred = np.load('../模型结果/ConvLSTM/20240701/reverse_pred_ConvLSTM_BoHai_sst.npy')
PredRNN_pred = np.load('../模型结果/PredRNN/20240701/reverse_pred_PredRNN_Bohai_sst.npy')
TCTN_pred = np.load('../模型结果/TCTN/20240701/reverse_pred_TCTN_Bohai_sst.npy')
SwinLSTM_pred = np.load('../模型结果/SwinLSTM/20240701/reverse_pred_SwinLSTM_BoHai_sst.npy')
DatLSTM_pred = np.load('../模型结果/DatLSTM/20240701/reverse_pred_DatLSTM_BoHai_sst.npy')



sample_index =1200

mask = np.load('../Mask/mask.npy')
true = true[sample_index, ...]
# PredRNN_pred = PredRNN_pred[sample_index, ...]
# ConvLSTM_pred = ConvLSTM_pred[sample_index, ::-1, ...]
# MoDeRNN_pred = MoDeRNN_pred[sample_index, ...]
# ST_GCFN_pred = ST_GCFN_pred[sample_index, ...]
# SwinLSTM_pred = SwinLSTM_pred[sample_index,...]
order = ['st', 'nd', 'rd', 'th','th', 'th','th', 'th','th', 'th']
# 创建包含四个子图的图像
fig, axes = plt.subplots(nrows=6, ncols=10, figsize=(16, 8), dpi=300)

vmax = true.max()
vmin = true[true != 0].min()
print(vmin)
print(vmax)
# 定义自定义颜色映射列表
# colors = ['#800000','#BD0202','#FE0002','#FF4103','#FF7D00','#FFBD00','#C0FE41','#81FF81','#41FEBE','#01FFFF','#02BDFE','#018DFB','#0000FD','#0001BD']
# colors = ['#FF3B00', '#FF7700', '#FF9400', '#FFB200', '#FEED00', '#CAFF2C', '#97FF60', '#63FF94', '#49FFAD', '#30FFC7', '#00E1FB', '#00A1FF', '#0061FF', '#0021FF']
colors = ['#FF0000','#FF3B00','#FF7700', '#FF9400', '#FFB200', '#FEED00', '#CAFF2C', '#97FF60',
'#63FF94', '#49FFAD', '#30FFC7', '#00E1FB', '#00A1FF', '#0061FF', '#0021FF']
# colors = ['#DC0000','#FF0000','#FF3B00', '#FF7700', '#FF9400', '#FFB200', '#FEED00', '#CAFF2C', '#97FF60',  '#49FFAD', '#30FFC7', '#00E1FB', '#00A1FF', '#0061FF', '#0021FF','#1D10CC']

# colors = ['#FF3B00', '#FF7700', '#FF9400', '#FEED00', '#CAFF2C', '#97FF60', '#30FFC7', '#00E1FB', '#00A1FF', '#0061FF', '#0021FF']

# colors = ['#800000',  '#FF0000','#DC143C','#CD5C5C','#F08080','#FA8072','#FF8C00','#FFA500','#FFD700','#FFFF00',
#           '#FFFFE0','#FFFACD','#90EE90','#008000','#008080','#40E0D0',
#           '#87CEEB','#00BFFF','#0000FF','#00008B']

# '#0000CD ','#006400','#B22222','#8B0000',

colors = colors[::-1]
# 绘制真实值子图
for i in range(10):
    sns.heatmap(np.flipud(true[i, ...]), ax=axes[0][i], cbar=False, mask = np.flipud(mask),cmap=colors, xticklabels=False, yticklabels=False, square=True, vmin=vmin, vmax=vmax)
    axes[0][i].set_title('$'+str(i+1)+'^{'+order[i]+'}$ day', y=1.1, fontdict={'family':'Times New Roman', 'size':15, 'weight': 'bold'})

    # 设置边框
    axes[0][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[0][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[0][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[0][i].spines['right'].set_visible(True)  # 显示底部轴线


# 绘制预测值子图
for i in range(10):

    # # ConvLSTM
    sns.heatmap(np.flipud(ConvLSTM_pred[sample_index, 9-i, ...]), ax=axes[1][i], mask = np.flipud(mask),cbar=False, cmap=colors, xticklabels=False,
                yticklabels=False,square=True, vmin=vmin, vmax=vmax)
    # 设置边框
    axes[1][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[1][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[1][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[1][i].spines['right'].set_visible(True)  # 显示底部轴线
    #
    # SwinLSTM
    sns.heatmap(np.flipud(PredRNN_pred[sample_index, i, ...]), ax=axes[2][i],mask = np.flipud(mask), cbar=False, cmap=colors, xticklabels=False,
                yticklabels=False, square=True, vmin=vmin, vmax=vmax)
    # 设置边框
    axes[2][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[2][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[2][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[2][i].spines['right'].set_visible(True)  # 显示底部轴线


    #ACFN
    sns.heatmap(np.flipud(TCTN_pred[sample_index, i, ...]), ax=axes[3][i], mask=np.flipud(mask), cbar=False,cmap=colors, xticklabels=False,
                yticklabels=False, square=True, vmin=vmin, vmax=vmax)
    # 设置边框
    axes[3][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[3][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[3][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[3][i].spines['right'].set_visible(True)  # 显示底部轴线

    # SE_ConcLSTM
    sns.heatmap(np.flipud(SwinLSTM_pred[sample_index, i, ...]), ax=axes[4][i], mask=np.flipud(mask), cbar=False,cmap=colors, xticklabels=False,
                yticklabels=False, square=True, vmin=vmin, vmax=vmax)
    # 设置边框
    axes[4][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[4][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[4][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[4][i].spines['right'].set_visible(True)  # 显示底部轴线
    # if i >= 4:
    #     temp = MoDeRNN_pred[i, ...]+0.5
    # else:
    #     temp  = MoDeRNN_pred[i, ...]
    #
    #

    # SK_ConcLSTM
    sns.heatmap(np.flipud(DatLSTM_pred[sample_index, i, ...]), ax=axes[5][i], mask=np.flipud(mask), cbar=False,cmap=colors, xticklabels=False,
                yticklabels=False, square=True, vmin=vmin, vmax=vmax)
    # 设置边框
    axes[5][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[5][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[5][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[5][i].spines['right'].set_visible(True)  # 显示底部轴线
    # sns.heatmap(np.flipud(SK_ConvLSTM), ax=axes[3][i], mask = np.flipud(mask), cbar=False, cmap=colors, xticklabels=False, yticklabels=False, square=True, vmin=vmin, vmax=vmax)
    # # 设置边框
    # axes[3][i].spines['left'].set_visible(True)  # 显示左边轴线
    # axes[3][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    # axes[3][i].spines['top'].set_visible(True)  # 显示顶部轴线
    # axes[3][i].spines['right'].set_visible(True)  # 显示底部轴线

    # # if i >= 2:
    # #     temp = ST_GCFN_pred[i, ...] - 0.8
    # # else:
    # #     temp  = ST_GCFN_pred[i, ...]
    # sns.heatmap(np.flipud(temp), ax=axes[4][i], mask=np.flipud(mask), cbar=False, cmap=colors,
    #             xticklabels=False, yticklabels=False, square=True, vmin=vmin, vmax=vmax)
    # # 设置边框
    # axes[4][i].spines['left'].set_visible(True)  # 显示左边轴线
    # axes[4][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    # axes[4][i].spines['top'].set_visible(True)  # 显示顶部轴线
    # axes[4][i].spines['right'].set_visible(True)  # 显示底部轴线



# 在每行的第一个子图中添加文本标签

axes[0][0].text(-0.8, 0.5, 'OSTIA', fontsize=16, ha='center', va='center', transform=axes[0][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'weight': 'bold'})
axes[1][0].text(-0.8, 0.5, 'ConvLSTM', fontsize=16, ha='center', va='center', transform=axes[1][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'weight': 'bold'})
axes[2][0].text(-0.8, 0.5, 'PredRNN', fontsize=16, ha='center', va='center', transform=axes[2][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'weight': 'bold'})
axes[3][0].text(-0.8, 0.5, 'TCTN', fontsize=16, ha='center', va='center', transform=axes[3][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'weight': 'bold'})
axes[4][0].text(-0.8, 0.5, 'SwinLSTM', fontsize=16, ha='center', va='center', transform=axes[4][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'weight': 'bold'})
axes[5][0].text(-0.95, 0.5, 'DatLSTM(Ours)', fontsize=16, ha='center', va='center', transform=axes[5][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'weight': 'bold'})



# 调整子图之间的距离
plt.subplots_adjust(wspace=0.1, hspace=0.25)


cbaxes = fig.add_axes([0.92, 0.2, 0.02, 0.6])
cbar = plt.colorbar(axes[0,0].imshow([[vmin, vmax]],
                  cmap=plt.get_cmap('rainbow')),cax=cbaxes,
                     orientation= 'vertical')
cbar.ax.set_position([0.92, 0.1, 0.02, 0.8])
# 设置颜色条刻度和标签
# # cbar_ticks = [0.30℃, 0.66℃, 1.02, 0.8,1.0,1.2, 1.4,1.6, 1.8, 2.0,2.2, 2.4, 2.6,2.8, 3.0]
# cbar_ticks = np.linspace(3, 18, 16).tolist()     #   原始  3.4，16.7
# char_ticks=[]
#
# # cbar.set_ticks([(x - vmin) / (vmax - vmin) for x in cbar_ticks])
# # cbar.set_ticklabels([f"{x:.1f}" for x in cbar_ticks])
# cbar.set_ticks(cbar_ticks)
# # cbar.ax.tick_params(labelsize=8)
# colorbar = fig.colorbar(axes[0][0].collections[0], ax=axes.ravel().tolist(), cax=cbaxes, shrink=0.60, aspect=10,ticks=cbar_ticks)
# cbar.set_ticklabels([f"{x:.2f}℃" for x in cbar_ticks])
# # colorbar.ax.tick_params(labelsize=20)  # 设置图例刻度标签的字体大小为12
# # colorbar.ax.yaxis.label.set_weight('bold')  # 设置标签加粗
# # # 添加共享的颜色条℃
# # cbar_ticks = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
# # # cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
# # cbar_ax = fig.add_axes([0.915, 0.12, 0.015, 0.75])
# # colorbar = fig.colorbar(axes[0][0].collections[0], ax=axes.ravel().tolist(), cax=cbar_ax, shrink=0.90, aspect=10)
# colorbar.ax.tick_params(labelsize=18)  # 设置图例刻度标签的字体大小为12
# colorbar.ax.yaxis.label.set_weight('bold')  # 设置标签加粗
#
ax = plt.gca()

# 在图像右侧添加垂直的数字标注
cbar = plt.colorbar(cax=ax.figure.add_axes([0.92, 0.1, 0.02, 0.8]), orientation='vertical')
cbar.set_label('Vertical Numerical\nDescription', rotation=270, labelpad=20)

# 设置刻度和标签
cbar.set_ticks(np.linspace(cbar.norm.vmin, cbar.norm.vmax, 5))
cbar.set_ticklabels(['{:.1f}'.format(x) for x in np.linspace(cbar.norm.vmin, cbar.norm.vmax, 5)])

# 调整坐标轴范围和标签
ax.set_xlim(0, 10)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('My Image')


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
plt.show()

