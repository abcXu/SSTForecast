import numpy as np
import numpy.ma as ma

mae = np.load('D:/Study/PredRNN/days_names_mae.npy')
rmse = np.load('D:/Study/PredRNN/days_names_rmse.npy')
r_2 = np.load('D:/Study/PredRNN/days_names_r_2.npy')
# rmse= np.load('D:/Study/SwinLSTM-main/数据分析结果/20240701/days_names_rmse.npy')
# mae= np.load('D:/Study/SwinLSTM-main/数据分析结果/20240701/days_names_mae.npy')
# r_2 = np.load('D:/Study/SwinLSTM-main/数据分析结果/20240701/days_names_r_2.npy')
# mae = np.load('../数据分析结果/20240701/days_mae_later.npy')
# rmse = np.load('../数据分析结果/20240701/days_rmse_later.npy')
# r_2 = np.load('../数据分析结果/20240701/days_r_2_later.npy')
# mae = np.load('../数据分析结果/20240701/days_mae_later.npy')
# rmse = np.load('../数据分析结果/20240701/days_rmse_later.npy')
# r_2 = np.load('../数据分析结果/20240701/days_r_2_later.npy')
#仅渤海区域有1565数据点
mask = np.load('D:/Study/PredRNN/PredRNN/maskLand=0.npy')

r_2_mask_broadcasted = np.broadcast_to(mask, (r_2 .shape[0], r_2 .shape[1], r_2 .shape[2], r_2 .shape[3]))
# 对于每个样本，使用masked_array函数将掩码为True的值替换为np.ma.masked
r_2_values_masked = ma.masked_array(r_2, r_2_mask_broadcasted)
r_2_masked_data = ma.filled(r_2_values_masked, 0)# 将掩码为True的数据置为0

nonzero_count = np.count_nonzero(r_2_masked_data, axis=(2, 3))  # 统计每个元素非0值的数量
nonzero_sum = np.sum(r_2_masked_data, axis=(2, 3))  # 统计每个元素非0值的总和
r_2_result = nonzero_sum / nonzero_count.astype(np.float32)  # 计算每个元素非0值的平均值
r_2_result = np.round(r_2_result, 3)

mae_mask_broadcasted = np.broadcast_to(mask, (mae.shape[0], mae.shape[1], mae.shape[2], mae.shape[3]))
# 对于每个样本，使用masked_array函数将掩码为True的值替换为np.ma.masked
mae_values_masked = ma.masked_array(mae, mae_mask_broadcasted)
mae_masked_data = ma.filled(mae_values_masked, 0)# 将掩码为True的数据置为0

nonzero_count = np.count_nonzero(mae_masked_data, axis=(2, 3))  # 统计每个元素非0值的数量
nonzero_sum = np.sum(mae_masked_data, axis=(2, 3))  # 统计每个元素非0值的总和
mae_result = nonzero_sum / nonzero_count.astype(np.float32)  # 计算每个元素非0值的平均值
mae_result = np.round(mae_result, 2)

rmse_mask_broadcasted = np.broadcast_to(mask, (rmse.shape[0], rmse.shape[1], rmse.shape[2], rmse.shape[3]))
# 对于每个样本，使用masked_array函数将掩码为True的值替换为np.ma.masked
rmse_values_masked = ma.masked_array(rmse, rmse_mask_broadcasted)
rmse_masked_data = ma.filled(rmse_values_masked, 0)# 将掩码为True的数据置为0

nonzero_count = np.count_nonzero(rmse_masked_data, axis=(2, 3))  # 统计每个元素非0值的数量
nonzero_sum = np.sum(rmse_masked_data, axis=(2, 3))  # 统计每个元素非0值的总和
rmse_result = nonzero_sum / nonzero_count.astype(np.float32)  # 计算每个元素非0值的平均值
rmse_result = np.round(rmse_result, 2)

# print('' )
# 输出 r_2_result
print("r_2_result:")
print(r_2_result)

# 输出 mae_result
print("mae_result:")
print(mae_result)

# 输出 rmse_result
print("rmse_result:")
print(rmse_result)





