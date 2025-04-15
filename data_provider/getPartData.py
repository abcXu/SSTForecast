import numpy as np
import matplotlib.pyplot as plt
data_path = 'D:/GraduationThesis/codes/preData/all_data'
sst_X_name = 'sst_X.npy'
sst_Y_name = 'sst_Y.npy'
ssh_X_name = 'ssh_X.npy'
ssh_Y_name = 'ssh_Y.npy'
print("load data...")
sst_X = np.load(data_path + '/' + sst_X_name)
sst_Y = np.load(data_path + '/' + sst_Y_name)
ssh_X = np.load(data_path + '/' + ssh_X_name)
ssh_Y = np.load(data_path + '/' + ssh_Y_name)


part_data_path = 'D:/GraduationThesis/codes/preData/part_data'
part_sst_X_name = 'sst_X.npy'
part_sst_Y_name = 'sst_Y.npy'
part_ssh_X_name = 'ssh_X.npy'
part_ssh_Y_name = 'ssh_Y.npy'

len = sst_X.shape[0]//10
#取完整数据的前10%用于测试
sst_X_part = sst_X[:len]
sst_Y_part = sst_Y[:len]
ssh_X_part = ssh_X[:len]
ssh_Y_part = ssh_Y[:len]

# 保存部分数据
np.save(part_data_path + '/' + part_sst_X_name,sst_X_part)
np.save(part_data_path + '/' + part_sst_Y_name,sst_Y_part)
np.save(part_data_path + '/' + part_ssh_X_name,ssh_X_part)
np.save(part_data_path + '/' + part_ssh_Y_name,ssh_Y_part)
print("data save finished!")