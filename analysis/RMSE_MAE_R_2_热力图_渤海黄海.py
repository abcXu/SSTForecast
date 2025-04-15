from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn import metrics

# name_list = ['ConvLSTM', 'PredRNN', 'TCTN', 'MoDeRNN', 'PhyDNet', 'SimVP', 'ACFN', 'DSTFN']

# name_list = ['ConvLSTM','PredRNN','TCTN','SwinLSTM','DatLSTM']
name_list = ['PredRNN']

def MAE(pred, true, day):
    #加载某一天的所有数据
    pred = pred[:, day, ...].reshape(-1, 64, 64)
    true = true[:, day, ...].reshape(-1, 64, 64)
    #计算某一天的所有数据均方误差之和
    day_mae = np.zeros((64, 64))
    for x in range(64):
        for y in range(64):
                day_mae[x, y] = mean_absolute_error(true[:, x, y], pred[:, x, y])
    return day_mae

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


mae_list_day = []
rmse_list_day = []
r_2_list_day = []

path = 'D:/Study/PredRNN/'
for day in range(0, 10):
    mae_list_name = []
    rmse_list_name  = []
    r_2_list_name  = []
    for name in name_list:
        # #加载归一化的数据
        # true_R = np.load(path + name + '/20240701/normalize_true_'+name+'_Bohai_sst.npy')
        # pred_R = np.load(path + name +'/20240701/normalize_pred_'+name+'_Bohai_sst.npy')

        # 加载逆归一化的数据
        # pred = np.load(path + name +'/20240701/reverse_pred_'+name+'_BoHai_sst.npy')
        # true = np.load(path + name +'/20240701/reverse_true_'+name+'_BoHai_sst.npy')
        pred_R = np.load(path + name + '/reverse_pred.npy')
        true_R = np.load(path + name + '/reverse_true.npy')
        # pred = np.load('./PredRNN/reverse_pred.npy')
        # true = np.load('./PredRNN/reverse_true.npy')
        if name == 'ConvLSTM':
            true_R = true_R[:, ::-1, :, :]
            pred_R = pred_R[:, ::-1, :, :]
            pred = pred[:, ::-1, :, :]
            true = true[:, ::-1, :, :]

        mae = MAE(pred_R, true_R, day)
        mae_list_name.append(mae)

        rmse = RMSE(pred_R, true_R, day)
        rmse_list_name.append(rmse)

        r_2 = R_2(pred_R, true_R, day)
        r_2_list_name.append(r_2)

    mae_list_day.append(mae_list_name)
    rmse_list_day.append(rmse_list_name)
    r_2_list_day.append(r_2_list_name)


np.save('D:/Study/PredRNN/days_names_rmse.npy', rmse_list_day)
np.save('D:/Study/PredRNN/days_names_mae.npy', mae_list_day)
np.save('D:/Study/PredRNN/days_names_r_2.npy', r_2_list_day)


rmse= np.load('D:/Study/PredRNN/days_names_rmse.npy')
mae= np.load('D:/Study/PredRNN/days_names_mae.npy')
r_2 = np.load('D:/Study/PredRNN/days_names_r_2.npy')



print(rmse.shape)  # (10,5,64,64)
print(mae.shape)   # (10,5,64,64)
print(r_2.shape)   # (10,5,64,64)
print(r_2)



# np.save('数据分析/days_names_rmse_heat(DSTFN-25).npy', rmse_list_day)
# np.save('数据分析/days_names_mae_heat(DSTFN-25).npy', mae_list_day)
# np.save('数据分析/days_names_r_2_heat(DSTFN-25).npy', r_2_list_day)
#
#
# rmse= np.load('数据分析/days_names_rmse_heat(DSTFN-25).npy')
# mae= np.load('数据分析/days_names_mae_heat(DSTFN-25).npy')
# r_2 = np.load('数据分析/days_names_r_2_heat(DSTFN-25).npy')

