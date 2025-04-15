import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# ['ConvLSTM','SwinLSTM','ACFN','SE_ConvLSTM','SK_ConvLSTM']

names = ['ConvLSTM', 'SwinLSTM', 'ACFN', 'SE_ConvLSTM', 'SK_ConvLSTM']
mask = np.load('G:/SwinLSTM-main/Mask/mask.npy')

true = np.load('../5个算法/ACFN/20240525/reverse_true_ACFN_Bohai_sst.npy').reshape(-1, 10, 64,64)


ConvLSTM_pred = np.load('../5个算法/ConvLSTM/20240525/reverse_pred_ConvLSTM_BoHai_sst.npy').reshape(-1, 10, 64,64)
SwinLSTM_pred = np.load('../5个算法/SwinLSTM/20240525/reverse_pred_SwinLSTM_BoHai_sst.npy').reshape(-1, 10, 64,64)
ACFN_pred = np.load('../5个算法/ACFN/20240525/reverse_pred_ACFN_Bohai_sst.npy').reshape(-1, 10, 64,64)

SE_ConvLSTM_pred = np.load('../5个算法/SE_ConvLSTM/20240525/reverse_pred_SE_ConvLSTM_Bohai_sst.npy').reshape(-1, 10, 64,64)
SK_ConvLSTM_pred = np.load('../5个算法/SK_ConvLSTM/20240525/reverse_pred_SK_ConvLSTM_Bohai_sst.npy').reshape(-1, 10, 64,64)
# SimVP_pred = np.load('F:/实验数据(考虑瑞年_何老师)/experiment_result/SimVP/reverse/reverse_pred_SimVP_Bohai_sst.npy').reshape(-1, 10, 64,64)
# ACFN_pred = np.load('F:/实验数据(考虑瑞年_何老师)/experiment_result/ACFN/reverse/reverse_pred_ACFN_Bohai_sst.npy').reshape(-1, 10, 64,64)
# DSTFN_pred = np.load('F:/实验数据(考虑瑞年_何老师)/experiment_result/DSTFN/reverse/reverse_pred_DSTFN_Bohai_sst.npy').reshape(-1, 10, 64,64)

sample_index = 35
true = true[sample_index, ...]

ConvLSTM_pred = ConvLSTM_pred[sample_index, ::-1, ...]
SwinLSTM_pred = SwinLSTM_pred[sample_index, ...]
ACFN_pred = ACFN_pred[sample_index, ...]
SE_ConvLSTM_pred = SE_ConvLSTM_pred[sample_index, ...]
SK_ConvLSTM_pred = SK_ConvLSTM_pred[sample_index, ...]
# SimVP_pred = SimVP_pred[sample_index, ...]
# ACFN_pred = ACFN_pred[sample_index, ...]
# DSTFN_pred = DSTFN_pred[sample_index, ...]

colors = ['#FF3B00','#FFB200', '#FEED00', '#CAFF2C', '#30FFC7', '#00E1FB', '#00A1FF', '#0021FF']
colors = colors[::-1]
# 创建包含四个子图的图像
fig, axes = plt.subplots(nrows=6, ncols=10,  figsize=(32, 28))
# 定义vmax vmin
vmax = true.max()
vmin = true[true != 0].min()
order = ['st', 'nd', 'rd', 'th','th', 'th','th', 'th','th', 'th']

# 绘制真实值子图
for i in range(10):
    sns.heatmap(np.flipud(true[i, ...]), ax=axes[0][i], cbar=False, mask = np.flipud(mask),cmap=colors, xticklabels=False, yticklabels=False, square=True, vmin=vmin, vmax=vmax)
    axes[0][i].set_title('$'+str(i+1)+'^{'+order[i]+'}$ day', y=1.0, fontdict={'family':'Times New Roman', 'size':22, 'weight': 'bold'})

    # 设置边框
    axes[0][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[0][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[0][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[0][i].spines['right'].set_visible(True)  # 显示底部轴线


# 绘制预测值子图
for i in range(10):

    sns.heatmap(np.flipud(ConvLSTM_pred[i, ...]), ax=axes[1][i], cbar=False, mask = np.flipud(mask), cmap=colors, xticklabels=False, yticklabels=False, square=True, vmin=vmin, vmax=vmax)
    # 设置边框
    axes[1][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[1][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[1][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[1][i].spines['right'].set_visible(True)  # 显示底部轴线


    sns.heatmap(np.flipud(SwinLSTM_pred[i, ...]), ax=axes[2][i], mask = np.flipud(mask), cbar=False, cmap=colors, xticklabels=False,
                yticklabels=False, square=True, vmin=vmin, vmax=vmax)
    # 设置边框
    axes[2][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[2][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[2][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[2][i].spines['right'].set_visible(True)  # 显示底部轴线

    # if i >= 4:
    #     temp = MoDeRNN_pred[i, ...]+0.5
    # else:
    #     temp  = MoDeRNN_pred[i, ...]

    sns.heatmap(np.flipud(ACFN_pred[i, ...]), ax=axes[3][i], mask = np.flipud(mask), cbar=False, cmap=colors, xticklabels=False, yticklabels=False, square=True, vmin=vmin, vmax=vmax)
    # 设置边框
    axes[3][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[3][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[3][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[3][i].spines['right'].set_visible(True)  # 显示底部轴线

    # if i >= 2:
    #     temp = ST_GCFN_pred[i, ...] - 0.8
    # else:
    #     temp  = ST_GCFN_pred[i, ...]
    sns.heatmap(np.flipud(SE_ConvLSTM_pred[i, ...]), ax=axes[4][i], mask=np.flipud(mask), cbar=False, cmap=colors,
                xticklabels=False, yticklabels=False, square=True, vmin=vmin, vmax=vmax)
    # 设置边框
    axes[4][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[4][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[4][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[4][i].spines['right'].set_visible(True)  # 显示底部轴线

    sns.heatmap(np.flipud(SK_ConvLSTM_pred[i, ...]), ax=axes[5][i], mask=np.flipud(mask), cbar=False, cmap=colors,
                xticklabels=False, yticklabels=False, square=True, vmin=vmin, vmax=vmax)
    # 设置边框
    axes[5][i].spines['left'].set_visible(True)  # 显示左边轴线
    axes[5][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    axes[5][i].spines['top'].set_visible(True)  # 显示顶部轴线
    axes[5][i].spines['right'].set_visible(True)  # 显示底部轴线

    # sns.heatmap(np.flipud(SimVP_pred[i, ...]), ax=axes[6][i], mask=np.flipud(mask), cbar=False, cmap=colors,
    #             xticklabels=False, yticklabels=False, square=True, vmin=vmin, vmax=vmax)
    # # 设置边框
    # axes[6][i].spines['left'].set_visible(True)  # 显示左边轴线
    # axes[6][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    # axes[6][i].spines['top'].set_visible(True)  # 显示顶部轴线
    # axes[6][i].spines['right'].set_visible(True)  # 显示底部轴线
    #
    # # if i >= 2:
    # #     temp = ACFN_pred[i, ...] - 0.2
    # # else:
    # #     temp  = ACFN_pred[i, ...]
    #
    # sns.heatmap(np.flipud(ACFN_pred[i, ...]), ax=axes[7][i], mask=np.flipud(mask), cbar=False, cmap=colors,
    #             xticklabels=False, yticklabels=False, square=True, vmin=vmin, vmax=vmax)
    # # 设置边框
    # axes[7][i].spines['left'].set_visible(True)  # 显示左边轴线
    # axes[7][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    # axes[7][i].spines['top'].set_visible(True)  # 显示顶部轴线
    # axes[7][i].spines['right'].set_visible(True)  # 显示底部轴线
    #
    # sns.heatmap(np.flipud(DSTFN_pred[i, ...]), ax=axes[8][i], mask=np.flipud(mask), cbar=False, cmap=colors,
    #             xticklabels=False, yticklabels=False, square=True, vmin=vmin, vmax=vmax)
    # # 设置边框
    # axes[8][i].spines['left'].set_visible(True)  # 显示左边轴线
    # axes[8][i].spines['bottom'].set_visible(True)  # 显示底部轴线
    # axes[8][i].spines['top'].set_visible(True)  # 显示顶部轴线
    # axes[8][i].spines['right'].set_visible(True)  # 显示底部轴线

# 在每行的第一个子图中添加文本标签

axes[0][0].text(-0.5, 0.5, 'OSTIA', size=22, ha='center', va='center', transform=axes[0][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'weight': 'bold'})
axes[1][0].text(-0.5, 0.5, 'ConvLSTM', size=22, ha='center', va='center', transform=axes[1][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'weight': 'bold'})
axes[2][0].text(-0.5, 0.5, 'SwinLSTM', size=22, ha='center', va='center', transform=axes[2][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'weight': 'bold'})
axes[3][0].text(-0.5, 0.5, 'ACFN', size=22, ha='center', va='center', transform=axes[3][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'weight': 'bold'})
axes[4][0].text(-0.5, 0.5, 'SE_ConvLSTM', size=22, ha='center', va='center', transform=axes[4][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'weight': 'bold'})
axes[5][0].text(-0.5, 0.5, 'SK_ConvLSTM', size=22, ha='center', va='center', transform=axes[5][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'weight': 'bold'})
# axes[6][0].text(-0.5, 0.5, 'SimVP', fontsize=22, ha='center', va='center', transform=axes[6][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'weight': 'bold'})
# axes[7][0].text(-0.5, 0.5, 'ACFN', fontsize=22, ha='center', va='center', transform=axes[7][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'weight': 'bold'})
# axes[8][0].text(-0.5, 0.5, 'DSTFN', fontsize=22, ha='center', va='center', transform=axes[8][0].transAxes, rotation=0, fontdict={'family':'Times New Roman', 'weight': 'bold'})


# 调整子图之间的距离
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# 添加共享的颜色条
cbar_ax = fig.add_axes([0.91, 0.12, 0.015, 0.75])
colorbar = fig.colorbar(axes[0][0].collections[0], ax=axes.ravel().tolist(), cax=cbar_ax, shrink=0.90, aspect=10)
colorbar.ax.tick_params(labelsize=20)  # 设置图例刻度标签的字体大小为12
# colorbar.ax.yaxis.label.set_size(14)   # 设置标签字体大小
colorbar.ax.yaxis.label.set_weight('bold')  # 设置标签加粗

plt.savefig('../5个算法结果/20240525/5个算法真实预测图-35.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0)

plt.show()
