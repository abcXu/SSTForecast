import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime

path = "D:/Datasets/result/20250415/reversed/"
dates = pd.date_range(start='2016-01-01', end='2020-12-31', freq='D')
true = np.load(path+'reverse_true_sst_SSTPredictor_mlp_SCS.npy').reshape(-1,10,64,64)
pred = np.load(path+'reverse_pred_sst_SSTPredictor_mlp_SCS.npy').reshape(-1,10,64,64)

# 创建五个子图，共享x轴
fig, axs = plt.subplots(10, 1, figsize=(20, 30), sharex=True)
# 确定数据的最小和最大日期
start_date = dates.min()
end_date = dates.max()
for i in range(10):
    true_values = true[213:, i, 42, 42]  # 假设有5组数据，每个子图一个
    predicted_values = pred[213:, i, 42, 42]
    print("len_true:", true_values.shape)
    print("len_pred:", predicted_values.shape)
    # 数据整理到DataFrame
    data = pd.DataFrame({
        "Date": dates,
        "True": true_values,
        "Predicted": predicted_values
    })

    # 计算每日的RMSE
    data['Daily_RMSE'] = np.sqrt((data['True'] - data['Predicted']) ** 2)
    axs[i].set_xlim(start_date, end_date)  # 调整x轴范围
    # 计算每月的RMSE
    data['YearMonth'] = data['Date'].dt.to_period('M')
    monthly_rmse = data.groupby('YearMonth')['Daily_RMSE'].mean().reset_index()
    monthly_rmse['YearMonth'] = monthly_rmse['YearMonth'].dt.to_timestamp()

    # 绘图
    axs[i].plot(data['Date'], data['Daily_RMSE'], label='Daily RMSE', alpha=0.75)
    axs[i].plot(monthly_rmse['YearMonth'], monthly_rmse['Daily_RMSE'], label='Monthly RMSE', alpha=1, color='red')
    axs[i].scatter(monthly_rmse['YearMonth'], monthly_rmse['Daily_RMSE'], color='red', zorder=5, s=10)

    axs[i].set_ylabel('RMSE (℃)', fontdict={'family': 'Times New Roman', 'size': 20, 'weight': 'bold'})
    axs[i].grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
    axs[i].tick_params(axis='y', labelsize=16)
    # 设置刻度标签的字体
    for label in axs[i].get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(24)  # 设置字体大小为14
        label.set_fontweight('bold')  # 加粗
    # 最后一个子图显示横坐标数据
    if i == 9:
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axs[i].xaxis.set_major_locator(mdates.YearLocator())
        axs[i].tick_params(axis='x', labelsize=18, rotation=0)
        for label in axs[i].get_xticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(25)  # 设置字体大小为14
            label.set_fontweight('bold')  # 加粗

    else:
        axs[i].tick_params(axis='x', labelbottom=False)  # 隐藏其它子图的x轴标签
        # 在每个子图上绘制图例
        axs[i].legend(
            loc='upper right',  # 图例位置
            frameon=True,  # 显示图例边框
            framealpha=1,  # 图例边框透明度
            edgecolor='black',  # 图例边框颜色
            prop={'family': 'Times New Roman', 'size': 19, 'weight': 'bold'}  # 图例字体属性
        )
# 假设axs[-1]是最后一个子图的轴对象
ax = axs[-1]  # 获取最后一个子图的轴

# 获取当前所有x轴的刻度位置和刻度标签
current_ticks = ax.get_xticks()
current_ticklabels = [tick.get_text() for tick in ax.get_xticklabels()]

# 将新的刻度位置和标签添加到列表中
# 假设您的数据是按日期排序的，可以直接添加新的日期
new_tick = mdates.date2num(datetime(2021, 1, 1))  # 转换新日期为matplotlib的内部格式
current_ticks = list(current_ticks) + [new_tick]  # 添加新刻度位置
current_ticklabels += ['2021-01']  # 添加新刻度标签

# 假设axs是您的子图数组
for i, ax in enumerate(axs):
    # 为每个子图设置标题
    ax.set_title('Daily Statistics 2016-2020: '+str(i+1)+'-day lead time', fontdict={'family': 'Times New Roman', 'size': 21, 'weight': 'bold'})

# 对于每个子图ax，设置四边都显示刻度
for ax in axs:
    # 设置刻度显示在四边
    ax.xaxis.set_ticks_position('both')  # X轴上下都显示刻度
    ax.yaxis.set_ticks_position('both')  # Y轴左右都显示刻度

    # 同时，您可以使用tick_params进一步细化刻度的样式和方向
    ax.tick_params(axis='x', which='both', direction='in')  # X轴刻度向内
    ax.tick_params(axis='y', which='both', direction='in')  # Y轴刻度向内
# 对于每个子图ax，设置刻度线变粗
for ax in axs:
    # 设置x轴和y轴刻度线变粗
    ax.tick_params(axis='x', which='both', width=2)  # X轴刻度线变粗
    ax.tick_params(axis='y', which='both', width=2)  # Y轴刻度线变粗

    # 如果需要，也可以继续设置其他属性，如刻度线的长度、颜色等
    ax.tick_params(axis='both', which='both', direction='in', length=6, color='black')


# 假设axs是您的子图数组
for ax in axs:
    # 设置边框线条粗细
    ax.spines['top'].set_linewidth(2)    # 上边框
    ax.spines['bottom'].set_linewidth(2) # 下边框
    ax.spines['left'].set_linewidth(2)   # 左边框
    ax.spines['right'].set_linewidth(2)  # 右边框


# 设置新的刻度位置和标签
ax.set_xticks(current_ticks)
ax.set_xticklabels(current_ticklabels, rotation=0)  # 如果需要，可以设置标签的旋转角度

# 重新设置x轴的主要格式器，以确保日期格式正确
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 自动调整子图布局
plt.tight_layout()
plt.savefig('./pictures/Daily-Monthly-RMSE_L1.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
