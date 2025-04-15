import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy.ma as ma

mae = np.load('../数据分析结果/20240701/days_names_r_2.npy')
mask = np.load('D:/Study/SwinLSTM-main/Mask/mask_bs.npy')# 仅渤海区域有1565数据点

mae_mask_broadcasted = np.broadcast_to(mask, (mae.shape[0], mae.shape[1], mae.shape[2], mae.shape[3]))
# 对于每个样本，使用masked_array函数将掩码为True的值替换为np.ma.masked
mae_values_masked = ma.masked_array(mae, mae_mask_broadcasted)
mae_masked_data = ma.filled(mae_values_masked, 0)# 将掩码为True的数据置为0

# 创建一个包含四个子图的画布，每个子图包含10个子子图
fig = plt.figure(figsize=(30, 30))
gs = GridSpec(5, 10, figure=fig, wspace=0.14, hspace=0.15)
order = ['st', 'nd', 'rd', 'th','th', 'th','th', 'th','th', 'th']

bins = np.arange(0, 4.000, 0.4)

bar_width = 0.4
fontdict = {'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
fontdict1 = {'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
# ['ConvLSTM','SwinLSTM','ACFN','SE_ConvLSTM','SK_ConvLSTM']
# 绘制ConvLSTM的Error分布
for i in range(10):
    ax = fig.add_subplot(gs[0, i])
    ax.axhline(y=200, color='black', linestyle='--', linewidth=1)
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    data = mae_masked_data[i, 0, ...]
    # 使用 np.nonzero() 函数获取非零元素的索引
    nonzero_indices = np.nonzero(data)
    # 提取非零元素
    mae_nonzero_elements = data[nonzero_indices]#1532

    temp = mae_nonzero_elements
    # print(1)   #4.970
    # max_value = max(temp)
    # print(max_value)
    # print(1)
    # 统计个数
    histogram, _ = np.histogram(temp, bins=bins)

    # 绘制柱状图，设置边框线颜色为黑色
    ax.bar(bins[:-1], histogram, width = bar_width, align='center', edgecolor='black', linewidth=1.5, color= '#D9582A')#

    ratio1 = np.sum(histogram[np.logical_and(bins[:-1] > 0, bins[:-1] <= 0.5)]) *100/ np.sum(histogram)
    ratio2 = np.sum(histogram[np.logical_and(bins[:-1] > 0.5, bins[:-1] <= 1)])*100 / np.sum(histogram)
    ratio3 = np.sum(histogram[np.logical_and(bins[:-1] > 1, bins[:-1] <= 1.5)]) *100/ np.sum(histogram)
    ratio4 = np.sum(histogram[bins[:-1] > 1.5])*100 / np.sum(histogram)
    # ratio4 = 100-ratio1-ratio2-ratio3

    ax.text(0.9, 0.9, f'p(0.0~0.5\u2103)={ratio1:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold', fontfamily='Times New Roman', va='top',
            ha='right', color = 'deepskyblue')
    ax.text(0.9, 0.8, f'p(0.5~1.0\u2103)={ratio2:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold', fontfamily='Times New Roman', va='top',
            ha='right', color = 'deepskyblue')
    ax.text(0.9, 0.7, f'p(1.0~1.5\u2103)={ratio3:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold', fontfamily='Times New Roman', va='top',
            ha='right', color = 'deepskyblue')
    ax.text(0.9, 0.6, f'p(>1.5\u2103)={ratio4:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold', fontfamily='Times New Roman', va='top',
            ha='right', color = 'deepskyblue')
    # # 设置y轴刻度范围
    # ax.set_ylim([0, 450])  # 设置合适的范围

    # 在第一个子图的左侧添加标题
    if(i==0):
        ax.text(-0.65, 0.5, 'ConvLSTM', fontdict=fontdict, ha='center', va='center', transform=ax.transAxes, rotation=0)
        ax.set_xticks([])
    else:
        ax.set_yticks([])
    ax.set_ylim([0, 1600])  # 设置合适的范围
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    plt.xticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')
    plt.yticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')


# 绘制TCTN的Error分布
for i in range(10):
    ax = fig.add_subplot(gs[1, i])
    ax.axhline(y=200, color='black', linestyle='--', linewidth=1)
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    data = mae_masked_data[i, 1, ...]
    # 使用 np.nonzero() 函数获取非零元素的索引
    nonzero_indices = np.nonzero(data)
    # 提取非零元素
    mae_nonzero_elements = data[nonzero_indices]
    temp = mae_nonzero_elements
    # print(1)  #1.75
    # max_value = max(temp)
    # print(max_value)
    # print(1)
    # 统计个数
    histogram, _ = np.histogram(temp, bins=bins)
    # 绘制柱状图，设置边框线颜色为黑色
    ax.bar(bins[:-1], histogram, width = bar_width, align='center', edgecolor='black', linewidth=1.5, color= '#D9582A')

    ratio1 = np.sum(histogram[np.logical_and(bins[:-1] > 0, bins[:-1] <= 0.5)])*100 / np.sum(histogram)
    ratio2 = np.sum(histogram[np.logical_and(bins[:-1] > 0.5, bins[:-1] <= 1)])*100 / np.sum(histogram)
    ratio3 = np.sum(histogram[np.logical_and(bins[:-1] > 1, bins[:-1] <= 1.5)]) *100/ np.sum(histogram)
    ratio4 = np.sum(histogram[bins[:-1] > 1.5]) / np.sum(histogram)
    # ratio4 = 100 - ratio1 - ratio2 - ratio3
    ax.text(0.9, 0.9, f'p(0.0~0.5\u2103)={ratio1:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold',
            fontfamily='Times New Roman', va='top',
            ha='right', color='deepskyblue')
    ax.text(0.9, 0.8, f'p(0.5~1.0\u2103)={ratio2:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold',
            fontfamily='Times New Roman', va='top',
            ha='right', color='deepskyblue')
    ax.text(0.9, 0.7, f'p(1.0~1.5\u2103)={ratio3:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold',
            fontfamily='Times New Roman', va='top',
            ha='right', color='deepskyblue')
    ax.text(0.9, 0.6, f'p(>1.5\u2103)={ratio4:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold',
            fontfamily='Times New Roman', va='top',
            ha='right', color='deepskyblue')

    # ax.set_ylim([0, 500])  # 设置合适的范围

    # 在第一个子图的左侧添加标题TCTN
    if(i==0):
        ax.text(-0.65, 0.5, 'PredRNN', fontdict=fontdict, ha='center', va='center', transform=ax.transAxes, rotation=0)
        ax.set_xticks([])
    else:
        ax.set_yticks([])
    ax.set_ylim([0, 1500])  # 设置合适的范围
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    plt.xticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')
    plt.yticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')


# 绘制SwinLSTM的Error分布
for i in range(10):
    ax = fig.add_subplot(gs[2, i])
    ax.axhline(y=200, color='black', linestyle='--', linewidth=1)
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    data = mae_masked_data[i, 2, ...]
    # 使用 np.nonzero() 函数获取非零元素的索引
    nonzero_indices = np.nonzero(data)
    # 提取非零元素
    mae_nonzero_elements = data[nonzero_indices]
    temp = mae_nonzero_elements
    # print(1)  #1.5
    # max_value = max(temp)
    # print(max_value)
    # print(1)
    # 统计个数
    histogram, _ = np.histogram(temp, bins=bins)
    # 绘制柱状图，设置边框线颜色为黑色
    ax.bar(bins[:-1], histogram, width = bar_width, align='center', edgecolor='black', linewidth=1.5, color= '#D9582A')
    ratio1 = np.sum(histogram[np.logical_and(bins[:-1] > 0, bins[:-1] <= 0.5)])*100 / np.sum(histogram)
    ratio2 = np.sum(histogram[np.logical_and(bins[:-1] > 0.5, bins[:-1] <= 1)])*100 / np.sum(histogram)
    ratio3 = np.sum(histogram[np.logical_and(bins[:-1] > 1, bins[:-1] <= 1.5)]) *100/ np.sum(histogram)
    ratio4 = np.sum(histogram[bins[:-1] > 1.5])*100 / np.sum(histogram)
    # ratio4 = 100 - ratio1 - ratio2 - ratio3

    ax.text(0.9, 0.9, f'p(0.0~0.5\u2103)={ratio1:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold', fontfamily='Times New Roman', va='top',
            ha='right', color = 'deepskyblue')
    ax.text(0.9, 0.8, f'p(0.5~1.0\u2103)={ratio2:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold', fontfamily='Times New Roman', va='top',
            ha='right', color = 'deepskyblue')
    ax.text(0.9, 0.7, f'p(1.0~1.5\u2103)={ratio3:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold', fontfamily='Times New Roman', va='top',
            ha='right', color = 'deepskyblue')
    ax.text(0.9, 0.6, f'p(>1.5\u2103)={ratio4:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold', fontfamily='Times New Roman', va='top',
            ha='right', color = 'deepskyblue')


    # 在第一个子图的左侧添加标题 PredRNN
    if(i==0):
        ax.text(-0.65, 0.5, 'TCTN', fontdict=fontdict, ha='center', va='center', transform=ax.transAxes, rotation=0)
        ax.set_xticks([])
    else:
        ax.set_yticks([])
    ax.set_ylim([0, 2912])  # 设置合适的范围
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    plt.xticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')
    plt.yticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')


# 绘制DatLSTM的Error分布
for i in range(10):
    ax = fig.add_subplot(gs[3, i])
    ax.axhline(y=200, color='black', linestyle='--', linewidth=1)
    # 设置子图的外边框线宽度为2
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)

    data = mae_masked_data[i, 3, ...]
    # 使用 np.nonzero() 函数获取非零元素的索引
    nonzero_indices = np.nonzero(data)
    # 提取非零元素
    mae_nonzero_elements = data[nonzero_indices]
    temp = mae_nonzero_elements
    # 统计个数
    histogram, _ = np.histogram(temp, bins=bins)
    # 绘制柱状图，设置边框线颜色为黑色
    ax.bar(bins[:-1], histogram, width = bar_width, align='center', edgecolor='black', linewidth=1.5,color= '#D9582A')

    ratio1 = np.sum(histogram[np.logical_and(bins[:-1] > 0, bins[:-1] <= 0.5)])*100 / np.sum(histogram)
    ratio2 = np.sum(histogram[np.logical_and(bins[:-1] > 0.5, bins[:-1] <= 1)])*100 / np.sum(histogram)
    ratio3 = np.sum(histogram[np.logical_and(bins[:-1] > 1, bins[:-1] <= 1.5)])*100 / np.sum(histogram)
    ratio4 = np.sum(histogram[bins[:-1] > 1.5])*100 / np.sum(histogram)
    # ratio4 = 100 - ratio1 - ratio2 - ratio3

    ax.text(0.9, 0.9, f'p(0.0~0.5\u2103)={ratio1:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold',
            fontfamily='Times New Roman', va='top',
            ha='right', color='deepskyblue')
    ax.text(0.9, 0.8, f'p(0.5~1.0\u2103)={ratio2:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold',
            fontfamily='Times New Roman', va='top',
            ha='right', color='deepskyblue')
    ax.text(0.9, 0.7, f'p(1.0~1.5\u2103)={ratio3:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold',
            fontfamily='Times New Roman', va='top',
            ha='right', color='deepskyblue')
    ax.text(0.9, 0.6, f'p(>1.5\u2103)={ratio4:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold',
            fontfamily='Times New Roman', va='top',
            ha='right', color='deepskyblue')

    ax.set_ylim([0, 1500])  # 设置合适的范围
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    # 在第一个子图的左侧添加标题 ST-GCFN
    if(i==0):
        ax.text(-0.65, 0.5, 'SwinLSTM', fontdict=fontdict, ha='center', va='center', transform=ax.transAxes, rotation=0)
    else:
        ax.set_yticks([])
    plt.xticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')
    plt.yticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')

# 绘制PhyDNet的Error分布
for i in range(10):
    ax = fig.add_subplot(gs[4, i])
    ax.axhline(y=200, color='black', linestyle='--', linewidth=1)
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    data = mae_masked_data[i, 4, ...]
    # 使用 np.nonzero() 函数获取非零元素的索引
    nonzero_indices = np.nonzero(data)
    # 提取非零元素
    mae_nonzero_elements = data[nonzero_indices]#1532

    temp = mae_nonzero_elements

    # 统计个数
    histogram, _ = np.histogram(temp, bins=bins)

    # 绘制柱状图，设置边框线颜色为黑色
    ax.bar(bins[:-1], histogram, width = bar_width, align='center', edgecolor='black', linewidth=1.5, color= '#D9582A')#

    ratio1 = np.sum(histogram[np.logical_and(bins[:-1] > 0, bins[:-1] <= 0.5)]) *100/ np.sum(histogram)
    ratio2 = np.sum(histogram[np.logical_and(bins[:-1] > 0.5, bins[:-1] <= 1)])*100 / np.sum(histogram)
    ratio3 = np.sum(histogram[np.logical_and(bins[:-1] > 1, bins[:-1] <= 1.5)]) *100/ np.sum(histogram)
    ratio4 = np.sum(histogram[bins[:-1] > 1.5])*100 / np.sum(histogram)
    # ratio4 = 100-ratio1-ratio2-ratio3

    ax.text(0.9, 0.9, f'p(0.0~0.5\u2103)={ratio1:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold', fontfamily='Times New Roman', va='top',
            ha='right', color = 'deepskyblue')
    ax.text(0.9, 0.75, f'p(0.5~1.0\u2103)={ratio2:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold', fontfamily='Times New Roman', va='top',
            ha='right', color = 'deepskyblue')
    ax.text(0.9, 0.60, f'p(1.0~1.5\u2103)={ratio3:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold', fontfamily='Times New Roman', va='top',
            ha='right', color = 'deepskyblue')
    ax.text(0.9, 0.45, f'p(>1.5\u2103)={ratio4:.2f}%', transform=ax.transAxes, fontsize=16, fontweight='bold', fontfamily='Times New Roman', va='top',
            ha='right', color = 'deepskyblue')
    # # 设置y轴刻度范围
    # ax.set_ylim([0, 450])  # 设置合适的范围
    # ax.set_title('$' + str(i + 1) + '^{' + order[i] + '}$ day', y=1.0, fontdict=fontdict1)
    # 在第一个子图的左侧添加标题
    if(i==0):
        ax.text(-0.65, 0.5, 'DatLSTM', fontdict=fontdict, ha='center', va='center', transform=ax.transAxes, rotation=0)
        ax.set_xticks([])
    else:
        ax.set_yticks([])
    ax.set_ylim([0, 1500])  # 设置合适的范围
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    plt.xticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')
    plt.yticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')




plt.savefig('../数据分析结果/20240701/R_2误差高斯分布.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig('../数据分析结果/20240701/R_2误差高斯分布.eps', format='eps',dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig('../数据分析结果/20240701/R_2误差高斯分布-eps-converted-to.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)

plt.show()
#
#
# # 绘制PhyDNet的Error分布
# for i in range(10):
#     ax = fig.add_subplot(gs[4, i])
#     ax.axhline(y=200, color='black', linestyle='--', linewidth=1)
#     ax.spines['top'].set_linewidth(1.2)
#     ax.spines['bottom'].set_linewidth(1.2)
#     ax.spines['left'].set_linewidth(1.2)
#     ax.spines['right'].set_linewidth(1.2)
#     data = mae_masked_data[i, 4, ...]
#     # 使用 np.nonzero() 函数获取非零元素的索引
#     nonzero_indices = np.nonzero(data)
#     # 提取非零元素
#     mae_nonzero_elements = data[nonzero_indices]#1532
#
#     temp = mae_nonzero_elements
#
#     # 统计个数
#     histogram, _ = np.histogram(temp, bins=bins)
#
#     # 绘制柱状图，设置边框线颜色为黑色
#     ax.bar(bins[:-1], histogram, width = bar_width, align='center', edgecolor='black', linewidth=1.5, color= '#D9582A')#
#
#     ratio1 = np.sum(histogram[np.logical_and(bins[:-1] > 0, bins[:-1] <= 0.5)]) *100/ np.sum(histogram)
#     ratio2 = np.sum(histogram[np.logical_and(bins[:-1] > 0.5, bins[:-1] <= 1)])*100 / np.sum(histogram)
#     ratio3 = np.sum(histogram[np.logical_and(bins[:-1] > 1, bins[:-1] <= 1.5)]) *100/ np.sum(histogram)
#     ratio4 = np.sum(histogram[bins[:-1] > 1.5])*100 / np.sum(histogram)
#     # ratio4 = 100-ratio1-ratio2-ratio3
#
#     ax.text(0.9, 0.9, f'p(0.0~0.5\u2103)={ratio1:.2f}%', transform=ax.transAxes, fontsize=13, fontweight='bold', fontfamily='Times New Roman', va='top',
#             ha='right', color = 'deepskyblue')
#     ax.text(0.9, 0.8, f'p(0.5~1.0\u2103)={ratio2:.2f}%', transform=ax.transAxes, fontsize=13, fontweight='bold', fontfamily='Times New Roman', va='top',
#             ha='right', color = 'deepskyblue')
#     ax.text(0.9, 0.7, f'p(1.0~1.5\u2103)={ratio3:.2f}%', transform=ax.transAxes, fontsize=13, fontweight='bold', fontfamily='Times New Roman', va='top',
#             ha='right', color = 'deepskyblue')
#     ax.text(0.9, 0.6, f'p(>1.5\u2103)={ratio4:.2f}%', transform=ax.transAxes, fontsize=13, fontweight='bold', fontfamily='Times New Roman', va='top',
#             ha='right', color = 'deepskyblue')
#     # # 设置y轴刻度范围
#     # ax.set_ylim([0, 450])  # 设置合适的范围
#     # ax.set_title('$' + str(i + 1) + '^{' + order[i] + '}$ day', y=1.0, fontdict=fontdict1)
#     # 在第一个子图的左侧添加标题
#     if(i==0):
#         ax.text(-0.65, 0.5, 'PhyDNet', fontdict=fontdict, ha='center', va='center', transform=ax.transAxes, rotation=0)
#         ax.set_xticks([])
#     else:
#         ax.set_yticks([])
#     ax.set_ylim([0, 700])  # 设置合适的范围
#     ax.set_xticks([0, 0.5, 1, 1.5, 2])
#     plt.xticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')
#     plt.yticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')
#
#
# # 绘制SimVP的Error分布
# for i in range(10):
#     ax = fig.add_subplot(gs[5, i])
#     ax.axhline(y=200, color='black', linestyle='--', linewidth=1)
#     ax.spines['top'].set_linewidth(1.2)
#     ax.spines['bottom'].set_linewidth(1.2)
#     ax.spines['left'].set_linewidth(1.2)
#     ax.spines['right'].set_linewidth(1.2)
#     data = mae_masked_data[i, 5, ...]
#     # 使用 np.nonzero() 函数获取非零元素的索引
#     nonzero_indices = np.nonzero(data)
#     # 提取非零元素
#     mae_nonzero_elements = data[nonzero_indices]#1532
#
#     temp = mae_nonzero_elements
#
#     # 统计个数
#     histogram, _ = np.histogram(temp, bins=bins)
#
#     # 绘制柱状图，设置边框线颜色为黑色
#     ax.bar(bins[:-1], histogram, width = bar_width, align='center', edgecolor='black', linewidth=1.5, color= '#D9582A')#
#
#     ratio1 = np.sum(histogram[np.logical_and(bins[:-1] > 0, bins[:-1] <= 0.5)]) *100/ np.sum(histogram)
#     ratio2 = np.sum(histogram[np.logical_and(bins[:-1] > 0.5, bins[:-1] <= 1)])*100 / np.sum(histogram)
#     ratio3 = np.sum(histogram[np.logical_and(bins[:-1] > 1, bins[:-1] <= 1.5)]) *100/ np.sum(histogram)
#     ratio4 = np.sum(histogram[bins[:-1] > 1.5])*100 / np.sum(histogram)
#     # ratio4 = 100-ratio1-ratio2-ratio3
#
#     ax.text(0.9, 0.9, f'p(0.0~0.5\u2103)={ratio1:.2f}%', transform=ax.transAxes, fontsize=13, fontweight='bold', fontfamily='Times New Roman', va='top',
#             ha='right', color = 'deepskyblue')
#     ax.text(0.9, 0.8, f'p(0.5~1.0\u2103)={ratio2:.2f}%', transform=ax.transAxes, fontsize=13, fontweight='bold', fontfamily='Times New Roman', va='top',
#             ha='right', color = 'deepskyblue')
#     ax.text(0.9, 0.7, f'p(1.0~1.5\u2103)={ratio3:.2f}%', transform=ax.transAxes, fontsize=13, fontweight='bold', fontfamily='Times New Roman', va='top',
#             ha='right', color = 'deepskyblue')
#     ax.text(0.9, 0.6, f'p(>1.5\u2103)={ratio4:.2f}%', transform=ax.transAxes, fontsize=13, fontweight='bold', fontfamily='Times New Roman', va='top',
#             ha='right', color = 'deepskyblue')
#     # # 设置y轴刻度范围
#     # ax.set_ylim([0, 450])  # 设置合适的范围
#     # ax.set_title('$' + str(i + 1) + '^{' + order[i] + '}$ day', y=1.0, fontdict=fontdict1)
#     # 在第一个子图的左侧添加标题
#     if(i==0):
#         ax.text(-0.65, 0.5, 'SimVP', fontdict=fontdict, ha='center', va='center', transform=ax.transAxes, rotation=0)
#         ax.set_xticks([])
#     else:
#         ax.set_yticks([])
#     ax.set_ylim([0, 700])  # 设置合适的范围
#     ax.set_xticks([0, 0.5, 1, 1.5, 2])
#     plt.xticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')
#     plt.yticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')
#
#
#
# # 绘制ACFN的Error分布
# for i in range(10):
#     ax = fig.add_subplot(gs[6, i])
#     ax.axhline(y=200, color='black', linestyle='--', linewidth=1)
#     ax.spines['top'].set_linewidth(1.2)
#     ax.spines['bottom'].set_linewidth(1.2)
#     ax.spines['left'].set_linewidth(1.2)
#     ax.spines['right'].set_linewidth(1.2)
#     data = mae_masked_data[i, 6, ...]
#     # 使用 np.nonzero() 函数获取非零元素的索引
#     nonzero_indices = np.nonzero(data)
#     # 提取非零元素
#     mae_nonzero_elements = data[nonzero_indices]#1532
#
#     temp = mae_nonzero_elements
#
#     # 统计个数
#     histogram, _ = np.histogram(temp, bins=bins)
#
#     # 绘制柱状图，设置边框线颜色为黑色
#     ax.bar(bins[:-1], histogram, width = bar_width, align='center', edgecolor='black', linewidth=1.5, color= '#D9582A')#
#
#     ratio1 = np.sum(histogram[np.logical_and(bins[:-1] > 0, bins[:-1] <= 0.5)]) *100/ np.sum(histogram)
#     ratio2 = np.sum(histogram[np.logical_and(bins[:-1] > 0.5, bins[:-1] <= 1)])*100 / np.sum(histogram)
#     ratio3 = np.sum(histogram[np.logical_and(bins[:-1] > 1, bins[:-1] <= 1.5)]) *100/ np.sum(histogram)
#     ratio4 = np.sum(histogram[bins[:-1] > 1.5])*100 / np.sum(histogram)
#     # ratio4 = 100-ratio1-ratio2-ratio3
#
#     ax.text(0.9, 0.9, f'p(0.0~0.5\u2103)={ratio1:.2f}%', transform=ax.transAxes, fontsize=13, fontweight='bold', fontfamily='Times New Roman', va='top',
#             ha='right', color = 'deepskyblue')
#     ax.text(0.9, 0.8, f'p(0.5~1.0\u2103)={ratio2:.2f}%', transform=ax.transAxes, fontsize=13, fontweight='bold', fontfamily='Times New Roman', va='top',
#             ha='right', color = 'deepskyblue')
#     ax.text(0.9, 0.7, f'p(1.0~1.5\u2103)={ratio3:.2f}%', transform=ax.transAxes, fontsize=13, fontweight='bold', fontfamily='Times New Roman', va='top',
#             ha='right', color = 'deepskyblue')
#     ax.text(0.9, 0.6, f'p(>1.5\u2103)={ratio4:.2f}%', transform=ax.transAxes, fontsize=13, fontweight='bold', fontfamily='Times New Roman', va='top',
#             ha='right', color = 'deepskyblue')
#     # # 设置y轴刻度范围
#     # ax.set_ylim([0, 450])  # 设置合适的范围
#     # ax.set_title('$' + str(i + 1) + '^{' + order[i] + '}$ day', y=1.0, fontdict=fontdict1)
#     # 在第一个子图的左侧添加标题
#     if(i==0):
#         ax.text(-0.65, 0.5, 'ACFN', fontdict=fontdict, ha='center', va='center', transform=ax.transAxes, rotation=0)
#         ax.set_xticks([])
#     else:
#         ax.set_yticks([])
#     ax.set_ylim([0, 700])  # 设置合适的范围
#     ax.set_xticks([0, 0.5, 1, 1.5, 2])
#     plt.xticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')
#     plt.yticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')
#
#
# # 绘制DSTFN的Error分布
# for i in range(10):
#     ax = fig.add_subplot(gs[7, i])
#     ax.axhline(y=200, color='black', linestyle='--', linewidth=1)
#     ax.spines['top'].set_linewidth(1.2)
#     ax.spines['bottom'].set_linewidth(1.2)
#     ax.spines['left'].set_linewidth(1.2)
#     ax.spines['right'].set_linewidth(1.2)
#     data = mae_masked_data[i, 7, ...]
#     # 使用 np.nonzero() 函数获取非零元素的索引
#     nonzero_indices = np.nonzero(data)
#     # 提取非零元素
#     mae_nonzero_elements = data[nonzero_indices]#1532
#
#     temp = mae_nonzero_elements
#
#     # 统计个数
#     histogram, _ = np.histogram(temp, bins=bins)
#
#     # 绘制柱状图，设置边框线颜色为黑色
#     ax.bar(bins[:-1], histogram, width = bar_width, align='center', edgecolor='black', linewidth=1.5, color= '#D9582A')#
#
#     ratio1 = np.sum(histogram[np.logical_and(bins[:-1] > 0, bins[:-1] <= 0.5)]) *100/ np.sum(histogram)
#     ratio2 = np.sum(histogram[np.logical_and(bins[:-1] > 0.5, bins[:-1] <= 1)])*100 / np.sum(histogram)
#     ratio3 = np.sum(histogram[np.logical_and(bins[:-1] > 1, bins[:-1] <= 1.5)]) *100/ np.sum(histogram)
#     ratio4 = np.sum(histogram[bins[:-1] > 1.5])*100 / np.sum(histogram)
#     # ratio4 = 100-ratio1-ratio2-ratio3
#
#     ax.text(0.9, 0.9, f'p(0.0~0.5\u2103)={ratio1:.2f}%', transform=ax.transAxes, fontsize=13, fontweight='bold', fontfamily='Times New Roman', va='top',
#             ha='right', color = 'deepskyblue')
#     ax.text(0.9, 0.8, f'p(0.5~1.0\u2103)={ratio2:.2f}%', transform=ax.transAxes, fontsize=13, fontweight='bold', fontfamily='Times New Roman', va='top',
#             ha='right', color = 'deepskyblue')
#     ax.text(0.9, 0.7, f'p(1.0~1.5\u2103)={ratio3:.2f}%', transform=ax.transAxes, fontsize=13, fontweight='bold', fontfamily='Times New Roman', va='top',
#             ha='right', color = 'deepskyblue')
#     ax.text(0.9, 0.6, f'p(>1.5\u2103)={ratio4:.2f}%', transform=ax.transAxes, fontsize=13, fontweight='bold', fontfamily='Times New Roman', va='top',
#             ha='right', color = 'deepskyblue')
#     # # 设置y轴刻度范围
#     # ax.set_ylim([0, 450])  # 设置合适的范围
#     # ax.set_title('$' + str(i + 1) + '^{' + order[i] + '}$ day', y=1.0, fontdict=fontdict1)
#     # 在第一个子图的左侧添加标题
#     if(i==0):
#         ax.text(-0.65, 0.5, 'DSTFN', fontdict=fontdict, ha='center', va='center', transform=ax.transAxes, rotation=0)
#         ax.set_xticks([])
#     else:
#         ax.set_yticks([])
#     ax.set_ylim([0, 700])  # 设置合适的范围
#     ax.set_xticks([0, 0.5, 1, 1.5, 2])
#     plt.xticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')
#     plt.yticks(fontsize=15, fontweight='bold', fontfamily='Times New Roman')




# plt.savefig('D:/PythonProject/aos2023_sic_bs/大论文代码/图片/mae_error_distribution.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()


