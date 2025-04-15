import numpy as np
import matplotlib.pyplot as plt

# 加载数据
ostia = np.load('../模型结果/SwinLSTM/20240701/reverse_true_SwinLSTM_BoHai_sst.npy')  # 原始数据
pred_rnn = np.load('../模型结果/PredRNN/20240701/reverse_pred_PredRNN_Bohai_sst.npy')  # PredRNN 预测数据
conv_lstm = np.load('../模型结果/ConvLSTM/20240701/reverse_pred_ConvLSTM_BoHai_sst.npy')  # ConvLSTM 预测数据
conv_gru = np.load('../模型结果/SwinLSTM/20240701/reverse_pred_SwinLSTM_BoHai_sst.npy')  # ConvGRU 预测数据
tctn = np.load('../模型结果/TCTN/20240701/reverse_pred_TCTN_Bohai_sst.npy')  # TCTN 预测数据
pann = np.load('../模型结果/DatLSTM/20240701/reverse_pred_Dat2912_new_file.npy')  # PANN（你的模型）预测数据

# 选择第一维度的索引
index = 0  # 可以选择从 0 到 9 的任意索引
dates = np.arange(np.datetime64('2019-01-01'), np.datetime64('2019-01-01') + np.timedelta64(10, 'D'))

# 提取数据
ostia_values = ostia[index, :, :, :].mean(axis=(1, 2))  # 计算每个时间步的平均值
pred_rnn_values = pred_rnn[index, :, :, :].mean(axis=(1, 2))
conv_lstm_values = conv_lstm[index, :, :, :].mean(axis=(1, 2))
conv_gru_values = conv_gru[index, :, :, :].mean(axis=(1, 2))
tctn_values = tctn[index, :, :, :].mean(axis=(1, 2))
pann_values = pann[index, :, :, :].mean(axis=(1, 2))

# 绘制主图
plt.figure(figsize=(12, 6))
plt.plot(dates, ostia_values, label='OSTIA', color='black')
plt.plot(dates, pred_rnn_values, label='PredRNN', linestyle='--', color='red')
plt.plot(dates, conv_lstm_values, label='ConvLSTM', linestyle='--', color='orange')
plt.plot(dates, conv_gru_values, label='ConvGRU', linestyle='--', color='green')
plt.plot(dates, tctn_values, label='TCTN', linestyle='--', color='blue')
plt.plot(dates, pann_values, label='PANN (Ours)', color='cyan')

# 添加图例
plt.legend()
plt.title('SST Prediction Comparison for Index {}'.format(index))
plt.xlabel('Date')
plt.ylabel('SST (°C)')
plt.xticks(rotation=45)

# 显示图形
plt.tight_layout()
plt.show()