import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy.ma as ma

rmse = np.load('../5个算法结果/20240525/days_names_rmse.npy')
mask = np.load('G:/SwinLSTM-main/Mask/mask_bs.npy')


rmse_mask_broadcasted = np.broadcast_to(mask, (rmse.shape[0], rmse.shape[1], rmse.shape[2], rmse.shape[3]))
# 对于每个样本，使用masked_array函数将掩码为True的值替换为np.ma.masked
rmse_values_masked = ma.masked_array(rmse, rmse_mask_broadcasted)
rmse_masked_data = ma.filled(rmse_values_masked, 0)# 将掩码为True的数据置为0

# 创建一个包含四个子图的画布，每个子图包含10个子子图
fig = plt.figure(figsize=(25, 11))
gs = GridSpec(5, 10, figure=fig, wspace=0.14, hspace=0.15)
order = ['st', 'nd', 'rd', 'th','th', 'th','th', 'th','th', 'th']
# 统计从-3到3之间每隔0.1间隔内的个数
bins = np.arange(0, 5.0001, 0.5)

bar_width = 0.5

fontdict = {'family': 'Times New Roman', 'size': 18, 'weight': 'bold'}
fontdict1 = {'family': 'Times New Roman', 'size': 19, 'weight': 'bold'}


# 绘制ConvLSTM的Error分布
for i in range(10):
    ax = fig.add_subplot(gs[0, i])
    ax.axhline(y=200, color='black', linestyle='--', linewidth=1)
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    data = rmse_masked_data[i, 0, ...]
    # print(rmse_masked_data)

    # 使用 np.nonzero() 函数获取非零元素的索引
    nonzero_indices = np.nonzero(data)
    # 提取非零元素
    mae_nonzero_elements = data[nonzero_indices]

    temp = mae_nonzero_elements

    # 统计个数
    histogram, _ = np.histogram(temp, bins=bins)

    # 绘制柱状图，设置边框线颜色为黑色
    ax.bar(bins[:-1], histogram, width = bar_width, align='center', edgecolor='black', linewidth=1.5, color= '#D9582A')#

    ratio1 = np.sum(histogram[np.logical_and(bins[:-1] >= 0, bins[:-1] < 0.5)])*100 / np.sum(histogram)
    ratio2 = np.sum(histogram[np.logical_and(bins[:-1] >= 0.5, bins[:-1] < 1)])*100 / np.sum(histogram)
    ratio3 = np.sum(histogram[np.logical_and(bins[:-1] >= 1, bins[:-1] < 1.5)]) *100/ np.sum(histogram)
    ratio4 = np.sum(histogram[np.logical_and(bins[:-1] >= 1.5, bins[:-1] < 2)]) *100/ np.sum(histogram)
    ratio5 = np.sum(histogram[np.logical_and(bins[:-1] >= 2, bins[:-1] < 2.5)]) *100/ np.sum(histogram)
    ratio6 = np.sum(histogram[np.logical_and(bins[:-1] >= 2.5, bins[:-1] < 3)]) *100/ np.sum(histogram)
    ratio7 = np.sum(histogram[bins[:-1] >= 3]) *100/ np.sum(histogram)

    ax.text(0.9, 0.9, f'p(0.0~0.5\u2103)={ratio1:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold', fontfamily='Times New Roman', va='top',ha='right', color = 'deepskyblue')
    ax.text(0.9, 0.8, f'p(0.5~1.0\u2103)={ratio2:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold', fontfamily='Times New Roman', va='top',ha='right', color = 'deepskyblue')
    ax.text(0.9, 0.7, f'p(1.0~1.5\u2103)={ratio3:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold', fontfamily='Times New Roman', va='top',ha='right', color = 'deepskyblue')
    ax.text(0.9, 0.6, f'p(1.5~2\u2103)={ratio4:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold', fontfamily='Times New Roman', va='top',ha='right', color = 'deepskyblue')
    ax.text(0.9, 0.5, f'p(2~2.5\u2103)={ratio5:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold', fontfamily='Times New Roman', va='top',ha='right', color = 'deepskyblue')
    ax.text(0.9, 0.4, f'p(2.5~3\u2103)={ratio6:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold', fontfamily='Times New Roman', va='top',ha='right', color = 'deepskyblue')
    ax.text(0.9, 0.3, f'p(>3\u2103)={ratio7:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold', fontfamily='Times New Roman', va='top',ha='right', color = 'deepskyblue')

    # # 设置y轴刻度范围
    # ax.set_ylim([0, 450])  # 设置合适的范围
    ax.set_title('$' + str(i + 1) + '^{' + order[i] + '}$ day', y=1.0, fontdict=fontdict1)
    # 在第一个子图的左侧添加标题
    if(i==0):
        ax.text(-0.65, 0.5, 'ConvLSTM', fontdict=fontdict, ha='center', va='center', transform=ax.transAxes, rotation=0)
        ax.set_xticks([])
    else:
        ax.set_yticks([])
    ax.set_ylim([0, 1570])  # 设置合适的范围
    # 计算每个柱状图的中心位置
    # tick_positions = bins[:-1] + bar_width / 2
    tick_positions = bins[:-1] - bar_width / 2

    # 设置x轴刻度位置
    ax.set_xticks(tick_positions)
    # ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_xticklabels([' ','0.5', ' ','1.5', ' ','2.5', ' ', '3.5',' ','4.5'])
    plt.xticks(fontsize=13, fontweight='bold', fontfamily='Times New Roman')
    plt.yticks(fontsize=13, fontweight='bold', fontfamily='Times New Roman')


# 绘制swinlstm的Error分布
for i in range(10):
    ax = fig.add_subplot(gs[1, i])
    ax.axhline(y=200, color='black', linestyle='--', linewidth=1)
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    data = rmse_masked_data[i, 1, ...]
    # print(data)
    # 使用 np.nonzero() 函数获取非零元素的索引
    nonzero_indices = np.nonzero(data)
    # 提取非零元素
    mae_nonzero_elements = data[nonzero_indices]
    temp = mae_nonzero_elements

    # 统计个数
    histogram, _ = np.histogram(temp, bins=bins)
    # 绘制柱状图，设置边框线颜色为黑色
    ax.bar(bins[:-1], histogram, width = bar_width, align='center', edgecolor='black', linewidth=1.5, color= '#D9582A')

    ratio1 = np.sum(histogram[np.logical_and(bins[:-1] >= 0, bins[:-1] < 0.5)])*100  / np.sum(histogram)
    ratio2 = np.sum(histogram[np.logical_and(bins[:-1] >= 0.5, bins[:-1] < 1)])*100  / np.sum(histogram)
    ratio3 = np.sum(histogram[np.logical_and(bins[:-1] >= 1, bins[:-1] < 1.5)])*100  / np.sum(histogram)
    ratio4 = np.sum(histogram[np.logical_and(bins[:-1] >= 1.5, bins[:-1] < 2)])*100  / np.sum(histogram)

    ratio5 = np.sum(histogram[bins[:-1] >= 2])*100  / np.sum(histogram)

    ax.text(0.9, 0.9, f'p(0.0~0.5\u2103)={ratio1:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold',
            fontfamily='Times New Roman', va='top',ha='right', color='deepskyblue')
    ax.text(0.9, 0.8, f'p(0.5~1.0\u2103)={ratio2:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold',
            fontfamily='Times New Roman', va='top',
            ha='right', color='deepskyblue')
    ax.text(0.9, 0.7, f'p(1.0~1.5\u2103)={ratio3:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold',
            fontfamily='Times New Roman', va='top',
            ha='right', color='deepskyblue')
    ax.text(0.9, 0.6, f'p(>1.5\u2103)={ratio4:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold',
            fontfamily='Times New Roman', va='top',
            ha='right', color='deepskyblue')
    ax.text(0.9, 0.5, f'p(>1.5\u2103)={ratio5:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold',
            fontfamily='Times New Roman', va='top',
            ha='right', color='deepskyblue')


    # 在第一个子图的左侧添加标题PredRNN# ['ConvLSTM','SwinLSTM','ACFN','SE_ConvLSTM','SK_ConvLSTM']
    if(i==0):
        ax.text(-0.65, 0.5, 'SwinLSTM', fontdict=fontdict, ha='center', va='center', transform=ax.transAxes, rotation=0)
        ax.set_xticks([])
    else:
        ax.set_yticks([])
    ax.set_ylim([0, 1570])  # 设置合适的范围
    # 计算每个柱状图的中心位置
    # tick_positions = bins[:-1] + bar_width / 2
    tick_positions = bins[:-1] - bar_width / 2

    # 设置x轴刻度位置
    ax.set_xticks(tick_positions)
    # ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_xticklabels([' ', '0.5', ' ', '1.5', ' ', '2.5', ' ', '3.5', ' ', '4.5'])
    plt.xticks(fontsize=13, fontweight='bold', fontfamily='Times New Roman')
    plt.yticks(fontsize=13, fontweight='bold', fontfamily='Times New Roman')

# 绘制AFCN的Error分布
for i in range(10):
    ax = fig.add_subplot(gs[2, i])
    ax.axhline(y=200, color='black', linestyle='--', linewidth=1)
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    data = rmse_masked_data[i, 2, ...]
    # 使用 np.nonzero() 函数获取非零元素的索引
    nonzero_indices = np.nonzero(data)
    # 提取非零元素
    mae_nonzero_elements = data[nonzero_indices]
    temp = mae_nonzero_elements

    # 统计个数
    histogram, _ = np.histogram(temp, bins=bins)
    # 绘制柱状图，设置边框线颜色为黑色
    #problem
    # print(bins[:-1])
    ax.bar(bins[:-1], histogram, width = bar_width, align='center', edgecolor='black', linewidth=1.5, color= '#D9582A')
    # ax.bar(bins[:-1], histogram)
    # print(histogram)
    ratio1 = np.sum(histogram[np.logical_and(bins[:-1] >= 0, bins[:-1] < 0.5)])*100  / np.sum(histogram)
    ratio2 = np.sum(histogram[np.logical_and(bins[:-1] >= 0.5, bins[:-1] < 1)])*100  / np.sum(histogram)
    ratio3 = np.sum(histogram[np.logical_and(bins[:-1] >= 1, bins[:-1] < 1.5)])*100  / np.sum(histogram)
    # ratio4 = np.sum(histogram[np.logical_and(bins[:-1] > 1.5, bins[:-1] <= 2)])*100  / np.sum(histogram)
    ratio4 = np.sum(histogram[bins[:-1] >= 1.5])*100  / np.sum(histogram)
    # print(11111111)
    # print(ratio1)
    # print(ratio2)
    # print(ratio3)
    # print(ratio4)
    # print(ratio5)
    ax.text(0.9, 0.9, f'p(0.0~0.5\u2103)={ratio1:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold',fontfamily='Times New Roman', va='top',ha='right', color='deepskyblue')
    ax.text(0.9, 0.8, f'p(0.5~1.0\u2103)={ratio2:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold',fontfamily='Times New Roman', va='top',ha='right', color='deepskyblue')
    ax.text(0.9, 0.7, f'p(1.0~1.5\u2103)={ratio3:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold',fontfamily='Times New Roman', va='top',ha='right', color='deepskyblue')
    ax.text(0.9, 0.6, f'p(>1.5\u2103)={ratio4:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold',fontfamily='Times New Roman', va='top',ha='right', color='deepskyblue')
    # ax.text(0.9, 0.5, f'p(>1.5\u2103)={ratio5:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold',fontfamily='Times New Roman', va='top',ha='right', color='deepskyblue')


    # 在第一个子图的左侧添加标题PredRNN# ['ConvLSTM','SwinLSTM','ACFN','SE_ConvLSTM','SK_ConvLSTM']
    if(i==0):
        ax.text(-0.65, 0.5, 'ACFN', fontdict=fontdict, ha='center', va='center', transform=ax.transAxes, rotation=0)
        ax.set_xticks([])
    else:
        ax.set_yticks([])
    ax.set_ylim([0, 1570])  # 设置合适的范围
    # 计算每个柱状图的中心位置
    tick_positions = bins[:-1] - bar_width / 2

    # 设置x轴刻度位置
    ax.set_xticks(tick_positions)
    # ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_xticklabels([' ', '0.5', ' ', '1.5', ' ', '2.5', ' ', '3.5', ' ', '4.5'])
    # ax.set_xticklabels([' ', '0.5', ' ', '1.5', ' ', '2.5', ' ', '3.5', ' ', '4.5'])
    plt.xticks(fontsize=13, fontweight='bold', fontfamily='Times New Roman')
    plt.yticks(fontsize=13, fontweight='bold', fontfamily='Times New Roman')

# 绘制SE_ConvLSTM的Error分布
for i in range(10):
    ax = fig.add_subplot(gs[3, i])
    ax.axhline(y=200, color='black', linestyle='--', linewidth=1)
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    data = rmse_masked_data[i, 3, ...]
    # 使用 np.nonzero() 函数获取非零元素的索引
    nonzero_indices = np.nonzero(data)
    # 提取非零元素
    mae_nonzero_elements = data[nonzero_indices]
    temp = mae_nonzero_elements

    # 统计个数
    histogram, _ = np.histogram(temp, bins=bins)
    # 绘制柱状图，设置边框线颜色为黑色
    #problem
    # print(bins[:-1])
    ax.bar(bins[:-1], histogram, width = bar_width, align='center', edgecolor='black', linewidth=1.5, color= '#D9582A')
    # ax.bar(bins[:-1], histogram)
    # print(histogram)
    ratio1 = np.sum(histogram[np.logical_and(bins[:-1] >= 0, bins[:-1] < 0.5)])*100  / np.sum(histogram)
    ratio2 = np.sum(histogram[np.logical_and(bins[:-1] >= 0.5, bins[:-1] < 1)])*100  / np.sum(histogram)
    ratio3 = np.sum(histogram[np.logical_and(bins[:-1] >= 1, bins[:-1] < 1.5)])*100  / np.sum(histogram)
    # ratio4 = np.sum(histogram[np.logical_and(bins[:-1] > 1.5, bins[:-1] <= 2)])*100  / np.sum(histogram)
    ratio4 = np.sum(histogram[bins[:-1] >= 1.5])*100  / np.sum(histogram)
    # print(11111111)
    # print(ratio1)
    # print(ratio2)
    # print(ratio3)
    # print(ratio4)
    # print(ratio5)
    ax.text(0.9, 0.9, f'p(0.0~0.5\u2103)={ratio1:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold',fontfamily='Times New Roman', va='top',ha='right', color='deepskyblue')
    ax.text(0.9, 0.8, f'p(0.5~1.0\u2103)={ratio2:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold',fontfamily='Times New Roman', va='top',ha='right', color='deepskyblue')
    ax.text(0.9, 0.7, f'p(1.0~1.5\u2103)={ratio3:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold',fontfamily='Times New Roman', va='top',ha='right', color='deepskyblue')
    ax.text(0.9, 0.6, f'p(>1.5\u2103)={ratio4:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold',fontfamily='Times New Roman', va='top',ha='right', color='deepskyblue')
    # ax.text(0.9, 0.5, f'p(>1.5\u2103)={ratio5:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold',fontfamily='Times New Roman', va='top',ha='right', color='deepskyblue')


    # 在第一个子图的左侧添加标题PredRNN# ['ConvLSTM','SwinLSTM','ACFN','SE_ConvLSTM','SK_ConvLSTM']
    if(i==0):
        ax.text(-0.65, 0.5, 'SwinLSTM', fontdict=fontdict, ha='center', va='center', transform=ax.transAxes, rotation=0)
        ax.set_xticks([])
    else:
        ax.set_yticks([])
    ax.set_ylim([0, 1570])  # 设置合适的范围
    # 计算每个柱状图的中心位置
    tick_positions = bins[:-1] - bar_width / 2

    # 设置x轴刻度位置
    ax.set_xticks(tick_positions)
    # ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_xticklabels([' ', '0.5', ' ', '1.5', ' ', '2.5', ' ', '3.5', ' ', '4.5'])
    # ax.set_xticklabels([' ', '0.5', ' ', '1.5', ' ', '2.5', ' ', '3.5', ' ', '4.5'])
    plt.xticks(fontsize=13, fontweight='bold', fontfamily='Times New Roman')
    plt.yticks(fontsize=13, fontweight='bold', fontfamily='Times New Roman')


# 绘制SK_ConvLSTM的Error分布
for i in range(10):
    ax = fig.add_subplot(gs[4, i])
    ax.axhline(y=200, color='black', linestyle='--', linewidth=1)
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    data = rmse_masked_data[i, 4, ...]
    # 使用 np.nonzero() 函数获取非零元素的索引
    nonzero_indices = np.nonzero(data)
    # 提取非零元素
    mae_nonzero_elements = data[nonzero_indices]
    temp = mae_nonzero_elements

    # 统计个数
    histogram, _ = np.histogram(temp, bins=bins)
    # 绘制柱状图，设置边框线颜色为黑色
    #problem
    # print(bins[:-1])
    ax.bar(bins[:-1], histogram, width = bar_width, align='center', edgecolor='black', linewidth=1.5, color= '#D9582A')
    # ax.bar(bins[:-1], histogram)
    # print(histogram)
    ratio1 = np.sum(histogram[np.logical_and(bins[:-1] >= 0, bins[:-1] < 0.5)])*100  / np.sum(histogram)
    ratio2 = np.sum(histogram[np.logical_and(bins[:-1] >= 0.5, bins[:-1] < 1)])*100  / np.sum(histogram)
    ratio3 = np.sum(histogram[np.logical_and(bins[:-1] >= 1, bins[:-1] < 1.5)])*100  / np.sum(histogram)
    # ratio4 = np.sum(histogram[np.logical_and(bins[:-1] > 1.5, bins[:-1] <= 2)])*100  / np.sum(histogram)
    ratio4 = np.sum(histogram[bins[:-1] >= 1.5])*100  / np.sum(histogram)
    # print(11111111)
    # print(ratio1)
    # print(ratio2)
    # print(ratio3)
    # print(ratio4)
    # print(ratio5)
    ax.text(0.9, 0.9, f'p(0.0~0.5\u2103)={ratio1:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold',fontfamily='Times New Roman', va='top',ha='right', color='deepskyblue')
    ax.text(0.9, 0.8, f'p(0.5~1.0\u2103)={ratio2:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold',fontfamily='Times New Roman', va='top',ha='right', color='deepskyblue')
    ax.text(0.9, 0.7, f'p(1.0~1.5\u2103)={ratio3:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold',fontfamily='Times New Roman', va='top',ha='right', color='deepskyblue')
    ax.text(0.9, 0.6, f'p(>1.5\u2103)={ratio4:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold',fontfamily='Times New Roman', va='top',ha='right', color='deepskyblue')
    # ax.text(0.9, 0.5, f'p(>1.5\u2103)={ratio5:.2f}%', transform=ax.transAxes, fontsize=11, fontweight='bold',fontfamily='Times New Roman', va='top',ha='right', color='deepskyblue')


    # 在第一个子图的左侧添加标题PredRNN# ['ConvLSTM','SwinLSTM','ACFN','SE_ConvLSTM','SK_ConvLSTM']
    if(i==0):
        ax.text(-0.65, 0.5, 'DatLSTM', fontdict=fontdict, ha='center', va='center', transform=ax.transAxes, rotation=0)
        ax.set_xticks([])
    else:
        ax.set_yticks([])
    ax.set_ylim([0, 1570])  # 设置合适的范围
    # 计算每个柱状图的中心位置
    tick_positions = bins[:-1] - bar_width / 2

    # 设置x轴刻度位置
    ax.set_xticks(tick_positions)
    # ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_xticklabels([' ', '0.5', ' ', '1.5', ' ', '2.5', ' ', '3.5', ' ', '4.5'])
    # ax.set_xticklabels([' ', '0.5', ' ', '1.5', ' ', '2.5', ' ', '3.5', ' ', '4.5'])
    plt.xticks(fontsize=13, fontweight='bold', fontfamily='Times New Roman')
    plt.yticks(fontsize=13, fontweight='bold', fontfamily='Times New Roman')



plt.savefig('../数据分析结果/20240701/rmse误差高斯分布图.eps', format='eps', dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig('../数据分析结果/20240701/rmse误差高斯分布图-eps-converted-to.pdf', format='pdf',  dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
# plt.savefig('../数据分析结果/20240701/MAE误差高斯分布.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0)
# plt.savefig('../数据分析结果/20240701/MAE误差高斯分布.eps', format='eps',dpi=300, bbox_inches='tight', pad_inches=0)
# plt.savefig('../数据分析结果/20240701/MAE误差高斯分布-eps-converted-to.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)



