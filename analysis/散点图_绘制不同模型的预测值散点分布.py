import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

from matplotlib.ticker import ScalarFormatter
import numpy.ma as ma
from sklearn.metrics import r2_score

# 加载数据
# convlstm_pred = np.load('../experiment_data/SST/ConvLSTM/reverse_normalize/reverse_pred_ConvLSTM_east2.npy').reshape(736, 10, 1, 64, 64)
# predrnn_pred = np.load('../experiment_data/SST/PredRNN/reverse_normalize/reverse_pred_PredRNN_east2.npy').reshape(736, 10, 1, 64, 64)
# tctn_pred = np.load('../experiment_data/SST/TCTN/reverse_normalize/reverse_pred_TCTN_east2.npy').reshape(736, 10, 1, 64, 64)
# convgru_pred = np.load('../experiment_data/SST/ConvGRU/reverse_normalize/reverse_pred_ConvGRU_east2.npy').reshape(736, 10, 1, 64, 64)
# swinlstm_pred = np.load('../experiment_data/SST/SwinLSTM/reverse_normalize/reverse_pred_swinLSTM_east2.npy').reshape(736, 10, 1, 64, 64)
# phy_att_conv_pred = np.load('../experiment_data/SST/Phy-Att-Conv/reverse_normalize/reverse_pred_Phy-Att-Conv_east2.npy').reshape(736, 10, 1, 64, 64)

convlstm_pred = np.load('../模型结果/ConvLSTM/20240701/reverse_pred_ConvLSTM_BoHai_sst.npy').reshape(2912, 10, 1, 64, 64)
predrnn_pred = np.load('../模型结果/PredRNN/20240701/reverse_pred_PredRNN_Bohai_sst.npy').reshape(2912, 10, 1, 64, 64)
tctn_pred = np.load('../模型结果/TCTN/20240701/reverse_pred_TCTN_Bohai_sst.npy').reshape(2912, 10, 1, 64, 64)
# convgru_pred = np.load('../experiment_data/SST/ConvGRU/reverse_normalize/reverse_pred_ConvGRU_east2.npy').reshape(736, 10, 1, 64, 64)
swinlstm_pred = np.load('../模型结果/SwinLSTM/20240701/reverse_pred_SwinLSTM_BoHai_sst.npy').reshape(2912, 10, 1, 64, 64)
datlstm_pred = np.load('../模型结果/DatLSTM/20240701/reverse_pred_Dat2912_new_file.npy').reshape(2912, 10, 1, 64, 64)
swinlstm_true = np.load('../模型结果/SwinLSTM/20240701/reverse_true_SwinLSTM_BoHai_sst.npy').reshape(2912, 10, 1, 64, 64)



# 引入真实数据
true = np.load('../模型结果/DatLSTM/20240701/reverse_true_Dat2912_new_file.npy').reshape(2912, 10, 1, 64, 64)


# 检查数据维度是否正确
# print('convlstm_pred.shape', convlstm_pred.shape)
# print('predrnn_pred.shape', predrnn_pred.shape)
# print('sa_convlstm_pred.shape', sa_convlstm_pred.shape)
# print('tctn_pred.shape', tctn_pred.shape)
# print('convgru_pred.shape', convgru_pred.shape)
# print('phy_att_conv_pred.shape', phy_att_conv_pred.shape)
#
# print('true.shape', true.shape)

# 所有数据维度一致 （736，10，1， 64，64）

print('数据加载完成')

# 定义分别存放 lead time = 1, 5, 10的预测值
convlstm_list = []
predrnn_list = []
tctn_list = []
convgru_list = []
swinlstm_list = []
phy_att_conv_list = []
true_list = []

# 分别取第1天、第5天、第10天的预测值
lead_time = [0, 4, 9]

# 循环获取对应 lead time 的预测值
for day in lead_time:
    # print(day)
    convlstm_list.append(convlstm_pred[:, 9-day, 0, :, :].flatten())
    predrnn_list.append(predrnn_pred[:, day, 0, :, :].flatten())
    tctn_list.append(tctn_pred[:, day, 0, :, :].flatten())
    convgru_list.append(swinlstm_true[:, day, 0, :, :].flatten())
    swinlstm_list.append(swinlstm_pred[:, day, 0, :, :].flatten())
    phy_att_conv_list.append(datlstm_pred[:, day, 0, :, :].flatten())
    true_list.append(true[:, day, 0, :, :].flatten())


# 每个算法四个评价指标，每个指标有10个值
metrics = np.zeros((5, 3, 10))

# ConvLSTM
metrics[0] = [
    [1.19, 1.20, 1.21, 1.21, 1.20, 1.17, 1.11, 1.10, 1.10, 1.71], # MAE
    [1.48, 1.49, 1.51, 1.51, 1.5, 1.45, 1.38, 1.37, 1.39, 2], # RMSE
    [0.973, 0.973, 0.972, 0.972, 0.972, 0.974, 0.976, 0.976, 0.976, 0.949], # R²
    # [0.947, 0.943, 0.941, 0.938, 0.937, 0.935, 0.933, 0.931, 0.929, 0.927]  # SSIM
]

# PredRNN
metrics[1] = [
    [1.09, 1.12, 1.15, 1.18, 1.21, 1.23, 1.25, 1.28, 1.30, 1.32],
    [1.38, 1.41, 1.44, 1.47, 1.51, 1.54, 1.57, 1.59, 1.62, 1.65],
    [0.977, 0.976, 0.975, 0.973, 0.972, 0.971, 0.970, 0.969, 0.968, 0.967],
    # [0.969, 0.968, 0.966, 0.965, 0.964, 0.963, 0.962, 0.961, 0.960, 0.959]
]


# TCTN
metrics[2] = [
    # [0.85, 0.90, 1.04, 1.21, 1.30, 1.42, 1.57, 1.80, 2.59, 3.12],
    # [0.96, 1.06, 1.29, 1.6, 1.92, 2.13, 2.33, 2.55, 2.86, 3.23],
    # [0.985, 0.978, 0.966, 0.952, 0.935, 0.915, 0.892, 0.864, 0.831, 0.793],
      [0.85,1.04 ,1.31 , 1.57 , 1.83 , 2.09 , 2.37 , 2.67 , 2.99 , 3.31 ],
	  [1.06 , 1.29 , 1.6 , 1.92 , 2.23 , 2.55 , 2.88 , 3.23 , 3.61 , 3.98],
	  [0.985, 0.978 , 0.966 , 0.952 , 0.935 , 0.915 , 0.892 , 0.864 , 0.831 , 0.793],

]

# # ConvGRU
# metrics[3] = [
#     [0.431, 0.513, 0.569, 0.624, 0.673, 0.718, 0.762, 0.805, 0.849, 0.895],
#     [0.635, 0.701, 0.76, 0.821, 0.878, 0.93, 0.982, 1.031, 1.081, 1.135],
#     [0.974, 0.971, 0.968, 0.963, 0.958, 0.954, 0.95, 0.945, 0.94, 0.935],
#     [0.977, 0.974, 0.972, 0.97, 0.967, 0.965, 0.963, 0.962, 0.96, 0.958]
# ]

# SwinLSTM
metrics[3] = [
    [0.61, 0.68, 0.73, 0.78, 0.83, 0.87, 0.91, 0.94, 0.98, 1.02],
    [0.81, 0.89, 0.96, 1.03, 1.08, 1.14, 1.19, 1.23, 1.28,1.32],
    [0.991, 0.989, 0.988, 0.986, 0.984, 0.983, 0.981, 0.98, 0.978, 0.977],
    # [0.984, 0.977, 0.974, 0.971, 0.969, 0.968, 0.967, 0.966, 0.965, 0.965]
]

# DatLSTM
metrics[4] = [
    [0.51, 0.62, 0.69, 0.74, 0.78, 0.81, 0.84, 0.87, 0.89, 0.91],
    [0.68, 0.84, 0.92, 0.98, 1.03, 1.07, 1.11, 1.14, 1.17, 1.2],
    [0.994, 0.991, 0.989, 0.987, 0.986, 0.985, 0.984, 0.983, 0.982, 0.981],
    # [0.987, 0.98, 0.975, 0.972, 0.97, 0.968, 0.967, 0.966, 0.966, 0.965]
]

# 设置camp类型
camp_color = 'jet' # coolwarm jet rainbow


# 创建一个包含5个子图的画布，每个子图包含3个子子图
fig = plt.figure(figsize=(13.5, 18))
gs = GridSpec(5, 3, figure=fig, wspace=0.0015, hspace=0.15)



# 配置参数
gridsize = 120
mincnt = 5
camp = 'jet' # coolwarm jet rainbow

# 共享颜色条
share_hb = None

# 绘制ConvLSTM的预测值散点分布
for i in range(3):
    ax = fig.add_subplot(gs[0, i])

    # 添加网格线
    ax.grid(True)

    ax.text(0.28, 0.90, 'RMSE={:.3f}'.format(metrics[0][1][lead_time[i]]), fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)
    ax.text(0.34, 0.78, 'R²={:.3f}'.format(metrics[0][2][lead_time[i]]), fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)

    pred = convlstm_list[i] # x 轴
    observed = convgru_list[i] # y 轴

    xlim = pred.min() - 2, pred.max() + 2     #########-2原始
    ylim = observed.min() - 2, observed.max() + 2

    hb = ax.hexbin(pred, observed, gridsize=gridsize, bins='log', cmap=camp, mincnt=mincnt, alpha=0.8)
    ax.set(xlim=ylim, ylim=ylim)

    y_major_locator = MultipleLocator(10)     ####原始为10
    ax.yaxis.set_major_locator(y_major_locator)

    x_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)
    # ax.set_xlim(0, 30)  # 将 x 轴刻度范围设置为 0 到 30

    ax.set_title(f"Day {lead_time[i] + 1}", y=1.05)
    # 设置图像标题
    # ax.text(0.5, -0.12, 'Predicted SST (℃)',  fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)

    # 在第一个子图的左侧添加标题
    if(i==0):
        share_hb = hb

        ax.text(-0.6, 0.5, 'ConvLSTM',  fontsize=12,  ha='center', va='center', transform=ax.transAxes, rotation=0)
        #
        ax.text(-0.2, 0.5, 'OSTIA (℃)', fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=90)

    if(i==2):
        cb = fig.colorbar(share_hb, ax=ax)

    # 设置子图为正方形
    ax.set_aspect('equal')





# 绘制PredRNN的预测值散点分布
for i in range(3):
    ax = fig.add_subplot(gs[1, i])
    
    # 添加网格线
    ax.grid(True)

    ax.text(0.28, 0.90, 'RMSE={:.3f}'.format(metrics[1][1][lead_time[i]]), fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)
    ax.text(0.34, 0.78, 'R²={:.3f}'.format(metrics[1][2][lead_time[i]]), fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)


    pred = predrnn_list[i] # x 轴
    observed = convgru_list[i] # y 轴

    xlim = pred.min() - 2, pred.max() + 2
    ylim = observed.min() - 2, observed.max() + 2

    hb = ax.hexbin(pred, observed, gridsize=gridsize, bins='log', cmap=camp, mincnt=mincnt, alpha=0.8)
    ax.set(xlim=ylim, ylim=ylim)

    # 设置x y 轴刻度分割为10
    y_major_locator = MultipleLocator(10)
    ax.yaxis.set_major_locator(y_major_locator)

    x_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)

    # 在第一个子图的左侧添加标题
    if(i==0):
        ax.text(-0.6, 0.5, 'PredRNN',  fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)
        ax.text(-0.2, 0.5, 'OSTIA (℃)', fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=90)

    if(i==2):
        cb = fig.colorbar(share_hb, ax=ax)

    # 设置子图为正方形
    ax.set_aspect('equal')



# 绘制TCTN的预测值散点分布
for i in range(3):
    ax = fig.add_subplot(gs[2, i])
    
    # 添加网格线
    ax.grid(True)

    ax.text(0.28, 0.90, 'RMSE={:.3f}'.format(metrics[2][1][lead_time[i]]), fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)
    ax.text(0.34, 0.78, 'R²={:.3f}'.format(metrics[2][2][lead_time[i]]), fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)


    pred = tctn_list[i] # x 轴
    observed = convgru_list[i] # y 轴

    xlim = pred.min() - 2, pred.max() + 2
    ylim = observed.min() - 2, observed.max() + 2

    hb = ax.hexbin(pred, observed, gridsize=gridsize, bins='log', cmap=camp, mincnt=mincnt, alpha=0.8)
    ax.set(xlim=ylim, ylim=ylim)

    # 设置x y 轴刻度分割为10
    y_major_locator = MultipleLocator(10)
    ax.yaxis.set_major_locator(y_major_locator)

    x_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)

    # 在第一个子图的左侧添加标题
    if(i==0):
        ax.text(-0.6, 0.5, 'TCTN',  fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)
        ax.text(-0.2, 0.5, 'OSTIA (℃)', fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=90)

    if(i==2):
        cb = fig.colorbar(share_hb, ax=ax)

    # 设置子图为正方形
    ax.set_aspect('equal')


# # 绘制ConvGRU的预测值散点分布
# for i in range(3):
#     ax = fig.add_subplot(gs[3, i])
#
#     # 添加网格线
#     ax.grid(True)
#
#     ax.text(0.28, 0.90, 'RMSE={:.3f}'.format(metrics[3][1][lead_time[i]]), fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)
#     ax.text(0.34, 0.78, 'R²={:.3f}'.format(metrics[3][2][lead_time[i]]), fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)
#
#
#     pred = convgru_list[i] # x 轴
#     observed = true_list[i] # y 轴
#
#     xlim = pred.min() - 2, pred.max() + 2
#     ylim = observed.min() - 2, observed.max() + 2
#
#     hb = ax.hexbin(pred, observed, gridsize=gridsize + 12, bins='log', cmap=camp, mincnt=mincnt+12, alpha=0.8)
#     ax.set(xlim=ylim, ylim=ylim)
#
#     # 设置x y 轴刻度分割为10
#     y_major_locator = MultipleLocator(10)
#     ax.yaxis.set_major_locator(y_major_locator)
#
#     x_major_locator = MultipleLocator(10)
#     ax.yaxis.set_major_locator(x_major_locator)
#
#     # 在第一个子图的左侧添加标题
#     if(i==0):
#         ax.text(-0.6, 0.5, 'SwinLSTM',  fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)
#         ax.text(-0.2, 0.5, 'OSTIA (℃)', fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=90)
#
#     if(i==2):
#         cb = fig.colorbar(share_hb, ax=ax)
#
#     # 设置子图为正方形
#     ax.set_aspect('equal')


# 绘制SwinLSTM
for i in range(3):
    ax = fig.add_subplot(gs[3, i])
    
    # 添加网格线
    ax.grid(True)

    ax.text(0.28, 0.90, 'RMSE={:.3f}'.format(metrics[3][1][lead_time[i]]), fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)
    ax.text(0.34, 0.78, 'R²={:.3f}'.format(metrics[3][2][lead_time[i]]), fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)


    pred = swinlstm_list[i] # x 轴
    observed = convgru_list[i] # y 轴

    xlim = pred.min() - 2, pred.max() + 2
    ylim = observed.min() - 2, observed.max() + 2

    hb = ax.hexbin(pred, observed, gridsize=gridsize, bins='log', cmap=camp, mincnt=mincnt, alpha=0.8)
    ax.set(xlim=ylim, ylim=ylim)

    # 设置x y 轴刻度分割为10
    y_major_locator = MultipleLocator(10)
    ax.yaxis.set_major_locator(y_major_locator)

    x_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)

    # 设置图像标题
    # ax.text(0.5, -0.18, 'Predicted SST (℃)',  fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)

    # 在第一个子图的左侧添加标题
    if(i==0):
        ax.text(-0.6, 0.5, 'SwinLSTM',  fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)
        ax.text(-0.18, 0.5, 'OSTIA (℃)', fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=90)

    if(i==2):
        cb = fig.colorbar(share_hb, ax=ax)

    # 设置子图为正方形
    ax.set_aspect('equal')

# 绘制PANN
for i in range(3):
    ax = fig.add_subplot(gs[4, i])

    # 添加网格线
    ax.grid(True)

    ax.text(0.28, 0.90, 'RMSE={:.3f}'.format(metrics[4][1][lead_time[i]]), fontsize=12, ha='center', va='center',
            transform=ax.transAxes, rotation=0)
    ax.text(0.34, 0.78, 'R²={:.3f}'.format(metrics[4][2][lead_time[i]]), fontsize=12, ha='center', va='center',
            transform=ax.transAxes, rotation=0)

    pred = phy_att_conv_list[i]  # x 轴
    observed = true_list[i]  # y 轴

    xlim = pred.min() - 2, pred.max() + 2
    ylim = observed.min() - 2, observed.max() + 2

    hb = ax.hexbin(pred, observed, gridsize=gridsize, bins='log', cmap=camp, mincnt=mincnt, alpha=0.8)
    ax.set(xlim=ylim, ylim=ylim)

    # 设置x y 轴刻度分割为10
    y_major_locator = MultipleLocator(10)
    ax.yaxis.set_major_locator(y_major_locator)

    x_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)

    # 设置图像标题
    ax.text(0.5, -0.18, 'Predicted SST (℃)', fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)

    # 在第一个子图的左侧添加标题
    if (i == 0):
        ax.text(-0.6, 0.5, 'DatLSTM', fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=0)
        ax.text(-0.18, 0.5, 'OSTIA (℃)', fontsize=12, ha='center', va='center', transform=ax.transAxes, rotation=90)

    if (i == 2):
        cb = fig.colorbar(share_hb, ax=ax)

    # 设置子图为正方形
    ax.set_aspect('equal')






# 添加共享颜色条
# cax = fig.add_axes([0.92, 0.118, 0.016, 0.755])
# fig.colorbar(ax, cax=cax, ticks=np.arange(vmin, vmax+0.1, 0.2), format='%0.1f'+'℃')
# fig.colorbar(hb, cax=cax)

# # 保存图片
plt.savefig('../数据分析结果/20240701/' + 'Fig9.png', dpi=300, bbox_inches='tight')

plt.savefig('../数据分析结果/20240701/' + 'Fig9.eps', dpi=300, bbox_inches='tight', format='eps')
# 显示图像
plt.show()
