import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import datetime
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, Shadow

path = "F:/实验数据(考虑瑞年_何老师)/experiment_result/DSTFN/reverse/"
dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')



true = np.load(path+'reverse_true_DSTFN_Bohai_sst.npy').reshape(-1,10,64,64)
pred = np.load(path+'reverse_pred_DSTFN_Bohai_sst.npy').reshape(-1,10,64,64)

lead = [2, 4, 6, 8, 10]

# 创建五个子图，共享x轴
fig, axs = plt.subplots(5, 1, figsize=(20, 20), sharex=True, constrained_layout=True)
# 确定数据的最小和最大日期
start_date = dates.min()
end_date = dates.max()
for i in range(0, 5):
    true_values = true[2196:2562, lead[i]-1, 23, 45]  # 假设有5组数据，每个子图一个
    predicted_values = pred[2196:2562, lead[i]-1, 23, 45]

    # 数据整理到DataFrame
    data = pd.DataFrame({
        "Date": dates,
        "True": true_values,
        "Predicted": predicted_values
    })

    # data['Daily_RMSE'] = np.sqrt((data['True'] - data['Predicted']) ** 2)
    data['Month'] = data['Date'].dt.to_period('M')
    error = abs(data['True'] - data['Predicted'])
    axs[i].plot(data['Date'], error, label='AE', alpha=1, linewidth = 3, color = 'orange')



    # 在每个子图的左上角标记"L1"并设置字体样式
    axs[i].text(0.02, 0.9, 'L3', transform=axs[i].transAxes, fontdict={
            'fontname': 'Times New Roman',
            'weight': 'bold',
            'fontsize': 25,
        }, verticalalignment='top', horizontalalignment='left')



    axs[i].set_ylabel('SST (℃)', fontdict={'family': 'Times New Roman', 'size': 24, 'weight': 'bold'})
    axs[i].grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
    axs[i].tick_params(axis='y', labelsize=16)
    # 设置刻度标签的字体
    for label in axs[i].get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(24)  # 设置字体大小为14
        label.set_fontweight('bold')  # 加粗
    # 最后一个子图显示横坐标数据
    if i == 4:
        # 设置x轴的主要刻度位置和格式
        axs[-1].xaxis.set_major_locator(mdates.MonthLocator())
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axs[-1].set_xlim([datetime.datetime(2020, 1, 1), datetime.datetime(2020, 12, 31)])
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
            prop={'family': 'Times New Roman', 'size': 23, 'weight': 'bold'}  # 图例字体属性
        )
# 假设axs[-1]是最后一个子图的轴对象
ax = axs[-1]  # 获取最后一个子图的轴

# 获取当前所有x轴的刻度位置和刻度标签
current_ticks = ax.get_xticks()
current_ticklabels = [tick.get_text() for tick in ax.get_xticklabels()]

# 将新的刻度位置和标签添加到列表中
# 假设您的数据是按日期排序的，可以直接添加新的日期
new_tick = mdates.date2num(datetime.datetime(2020, 1, 1))  # 转换新日期为matplotlib的内部格式
current_ticks = list(current_ticks) + [new_tick]  # 添加新刻度位置
current_ticklabels += ['2020-01']  # 添加新刻度标签

# 假设axs是您的子图数组
for i, ax in enumerate(axs):
    # 为每个子图设置标题
    ax.set_title(str(lead[i])+'-day lead time', fontdict={'family': 'Times New Roman', 'size': 24, 'weight': 'bold'})

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


#手动添加最后一个刻度为2021-01
last_tick_date = mdates.num2date(axs[-1].get_xticks()[-1]) + pd.DateOffset(months=1)  # 转换为datetime对象并加一个月
last_tick_num = mdates.date2num(last_tick_date)  # 转换回matplotlib日期数值
axs[-1].set_xticks(list(axs[-1].get_xticks()) + [last_tick_num])  # 添加新刻度

# 更新所有刻度标签，包括新添加的2021-01
all_ticks_num = axs[-1].get_xticks()  # 获取所有刻度（包括新添加的）
all_ticks_date = [mdates.num2date(t) for t in all_ticks_num]  # 转换为datetime对象
axs[-1].set_xticklabels([t.strftime('%Y-%m') for t in all_ticks_date])  # 设置新的刻度标签
# 自动调整子图布局
plt.tight_layout()
plt.savefig('D:/PythonProject/aos2023_sic_bs/大论文代码/图片/AE.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()


