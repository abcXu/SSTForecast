import numpy as np
model_name = 'SSTPredictor'
data_true_sst = np.load(f'../results/predict/20250414/normalized_true_sst_{model_name}_SCS.npy')
data_pred_sst = np.load(f'../results/predict/20250414/normalized_pred_sst_{model_name}_SCS.npy')
# data_true_ssh = np.load('../results/predict/20250411/normalize_true_DatLSTM_csc_ssh.npy')
# data_pred_ssh = np.load('../results/predict/20250411/normalize_pred_DatLSTM_csc_ssh.npy')
# 检查数组的形状
# print(data.shape)
# print(data)
# savePath = '../results/predict/20240422/'
savePath = 'D:/Datasets/result/20250414/'
# # # 归一化
# # def SST_minmaxscaler(data):
# #     max = 33.44
# #     min = -2.0
# #
# #     return (data - min) / (max - min)

# ##逆归一化 sst_boHai
# def SST_reverse_minmaxscaler(data):
#     max = 33.44
#     min = -2.0
#     return data * (max - min) + min

# scs sst_reverse
def SST_reverse_minmaxscaler(norm_data):

    max = 32.86
    min = 9.76

    return norm_data * (max - min) + min

def SSH_reverse_minmaxscaler(norm_data):

    max = 0.6699
    min = -0.6771

    return norm_data * (max - min) + min
# true = np.load(r'E:\Students-21\FengLiu\Deep_Learning_Code\SST_forecasting_FengLiu\experiments\20230328140957-ConvLSTM-high\normalize_true_ConvLSTM_east.npy')
#
# true1 = SST_reverse_minmaxscaler(area='highLat', norm_data=true)
#
# true2 = np.load(r'E:\Students-21\HaoYingJian\2022_07_17_SimVP\test_reverse_data\20230328142315\Normalize_true.npy')
#
# print(true1 == true2)

#
# print("正在进行归一化......")
# Normalize_sst = SST_reverse_minmaxscaler(data)
print("正在逆归一化.......")
reversed_sst_true = SST_reverse_minmaxscaler(data_true_sst)
reversed_sst_pred = SST_reverse_minmaxscaler(data_pred_sst)
# reversed_ssh_true = SSH_reverse_minmaxscaler(data_true_ssh)
# reversed_ssh_pred = SSH_reverse_minmaxscaler(data_pred_ssh)
print("正在逆归一化完成.......")
# Normalize_sst = SST_minmaxscaler(data)
# #
#
# print("归一化完成，正在将归一化后的数据保存为npy文件")
# np.save(savePath + 'reverse -')
# np.save(savePath + 'reverse-BohaiSea_50epoch_true.npy', Normalize_sst)
np.save(savePath + f'reverse_true_sst_{model_name}_SCS.npy', reversed_sst_true)
np.save(savePath + f'reverse_pred_sst_{model_name}_SCS.npy', reversed_sst_pred)
# np.save(savePath + 'reverse_true_ssh_DatLSTM_scs.npy', reversed_ssh_true)
# np.save(savePath + 'reverse_pred_ssh_DatLSTM_scs.npy', reversed_ssh_pred)
print()
print('保存成功！')
print("保存到"+savePath)
