from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn import metrics


# true_values = np.load('../模型结果/DatLSTM/20240701/reverse_true_DatLSTM_BoHai_sst.npy')
# predicted_values = np.load('../模型结果/DatLSTM/20240701/reverse_pred_DatLSTM_BoHai_sst.npy')


def R_2(pred, true, day):
    # 加载某一天的所有数据
    pred = pred[:, day, ...].reshape(-1, 64, 64)
    true = true[:, day, ...].reshape(-1, 64, 64)
    # 计算某一天的所有数据均方误差之和
    day_r_2 = np.zeros((64, 64))
    for x in range(64):
        for y in range(64):
            day_r_2[x, y] = metrics.r2_score(true[:, x, y], pred[:, x, y])
    return day_r_2


def RMSE(pred, true, day):
    #用于保所有样本第day天的平均RMSE
    day_rmse = np.zeros((64, 64))
    #加载某一天的所有数据
    pred = pred[:, day, ...].reshape(-1, 64, 64)
    true = true[:, day, ...].reshape(-1, 64, 64)
    #计算某一天的所有数据均方误差之和
    for x in range(64):
        for y in range(64):
            day_rmse[x, y] = np.sqrt(mean_squared_error(true[:, x, y], pred[:, x, y]))
    return day_rmse


rmse_list_day = []
r_2_list_day = []


for day in range(0, 10):
    # rmse_list_name  = []
    # r_2_list_name  = []
        # #加载归一化的数据
        # true_R = np.load(path + name + '/20240701/normalize_true_'+name+'_Bohai_sst.npy')
        # pred_R = np.load(path + name +'/20240701/normalize_pred_'+name+'_Bohai_sst.npy')
        #
        # # 加载逆归一化的数据
        # pred = np.load(path + name +'/20240701/reverse_pred_'+name+'_BoHai_sst.npy')
        # true = np.load(path + name +'/20240701/reverse_true_'+name+'_BoHai_sst.npy')

    true_R = np.load('../模型结果/DatLSTM/20240701/normalize_true_DatLSTM_Bohai_sst.npy')
    pred_R = np.load('../模型结果/DatLSTM/20240701/normalize_pred_DatLSTM_Bohai_sst.npy')

    # 加载逆归一化的数据
    pred = np.load('../模型结果/DatLSTM/20240701/reverse_pred_DatLSTM_BoHai_sst.npy')
    true = np.load('../模型结果/DatLSTM/20240701/reverse_true_DatLSTM_BoHai_sst.npy')

    rmse = RMSE(pred, true, day)
    # rmse_list_name.append(rmse)

    r_2 = R_2(pred_R, true_R, day)
    # r_2_list_name.append(r_2)

    rmse_list_day.append(rmse)
    r_2_list_day.append(r_2)


np.save('../模型结果/DatLSTM/20240701/datlstm_rmse.npy', rmse_list_day)
# # np.save('D:/Study/SwinLSTM-main/数据分析结果/20240701/days_names_mae.npy', mae_list_day)
np.save('../模型结果/DatLSTM/20240701/datlstm_r_2.npy', r_2_list_day)
#
#
rmse= np.load('../模型结果/DatLSTM/20240701/datlstm_rmse.npy')
# # mae= np.load('D:/Study/SwinLSTM-main/数据分析结果/20240701/days_names_mae.npy')
r_2 = np.load('../模型结果/DatLSTM/20240701/datlstm_r_2.npy')


print(rmse)
print(rmse.shape)  # (10,5,64,64)
print(r_2)   # (10,5,64,64)
print(r_2.shape)   # (10,5,64,64)



import numpy as np

# 加载 R^2 和 RMSE 数据
# r2_data = np.load('r2_data.npy')
# rmse_data = np.load('rmse_data.npy')

# 设置筛选条件
r2_threshold = 0.985
rmse_threshold = 0.95

# 找出满足条件的共同网格区域
good_mask = (r_2 > r2_threshold) & (rmse < rmse_threshold)

# 保存筛选后的共同网格区域为 npy 文件
np.save('../数据分析结果/20240701/good_mask.npy', good_mask)

# print(f"Saved the common grid region where R-squared > {r2_threshold} and RMSE < {rmse_threshold} to 'good_mask.npy'.")
print(good_mask)
print(good_mask.shape)