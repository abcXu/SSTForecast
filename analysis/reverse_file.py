import os

import numpy as np
model_name = 'SSTPredictor'
data_true_sst = np.load(f'../results/predict/20250506/normalized_true_sst_{model_name}_SCS.npy')
data_pred_sst = np.load(f'../results/predict/20250506/normalized_pred_sst_{model_name}_SCS.npy')

savePath = 'D:/Datasets/result/20250506/reversed/'
if not os.path.exists(savePath):
    os.makedirs(savePath)
def SST_reverse_minmaxscaler(norm_data):

    max_val = 32.86
    min_val = 9.76

    return norm_data * (max_val - min_val) + min_val

def SSH_reverse_minmaxscaler(norm_data):

    max_val = 0.6699
    min_val = -0.6771

    return norm_data * (max_val - min_val) + min_val

print("正在逆归一化.......")
reversed_sst_true = SST_reverse_minmaxscaler(data_true_sst)
reversed_sst_pred = SST_reverse_minmaxscaler(data_pred_sst)
print("正在逆归一化完成.......")
np.save(savePath + f'reverse_true_sst_{model_name}_mlp_SCS.npy', reversed_sst_true)
np.save(savePath + f'reverse_pred_sst_{model_name}_mlp_SCS.npy', reversed_sst_pred)
print('保存成功！')
print("保存到"+savePath)
