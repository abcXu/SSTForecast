# import numpy as np
# import matplotlib.pyplot as plt
#
# # 加载npy文件
# data = np.load('D:/Study\模型结果/DatLSTM/20240701/reverse_true_DatLSTM_BoHai_sst.npy')  # 请将 'your_file.npy' 替换为实际文件名
# print(data.shape)
# # 假设要获取的位置点在64x64空间维度中的索引为 (x_index, y_index)
# # 这里只是示例值，你需要根据实际情况确定
# x_index = 25
# y_index = 25
#
# # 假设时间跨度为两个月（60天左右），这里取前60个时间点的数据
# time_steps = 90
# selected_data = data[:time_steps, :, x_index, y_index]
#
# # 计算每个时间步长的平均温度
# average_temperatures = np.mean(selected_data, axis=1)
#
# # 生成时间序列
# time = np.arange(time_steps)
#
# # 绘制图形
# plt.plot(time, average_temperatures, label='海洋位置点')
# plt.xlabel('时间/天',horizontalalignment='center', verticalalignment='bottom', fontsize=12)
# plt.ylabel('海洋表面温度/摄氏度',horizontalalignment='center', verticalalignment='bottom', fontsize=12)
# plt.title('海洋位置点(31.125° N，24.375° E)在两个月内的海洋表面温度变化趋势曲线',horizontalalignment='center', verticalalignment='bottom', fontsize=12)
# plt.text(47.8, 51.5, '9', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
#
# # plt.legend()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载npy文件，需将文件名替换为实际的文件名
# data = np.load('D:/Study\模型结果/DatLSTM/20240701/reverse_true_DatLSTM_BoHai_sst.npy')
# data = np.load('D:/Datasets/result/20250415/reversed/reverse_true_sst_SSTPredictor_mlp_SCS.npy')
# data = np.load('D:/GraduationThesis/codes/preData/all_data/sst_X.npy').squeeze()
data = np.load('D:/Datasets/result/20250415/reversed/reverse_true_sst_SSTPredictor_mlp_SCS.npy ')
# 假设（30, 30）为64x64空间维度中的坐标索引
x_index, y_index = 30, 30

# 提取该点在所有14591天的温度数据
time_series_temperature = data[:, :, x_index, y_index].mean(axis=1)

# 生成时间序列，从1到14591
time_days = np.arange(1, data.shape[0] + 1)

# 绘制温度变化曲线

plt.plot(time_days, time_series_temperature)
plt.xlabel('天数')
plt.ylabel('温度/摄氏度')
plt.title(f'坐标点({x_index}, {y_index})处的温度变化曲线')
plt.show()