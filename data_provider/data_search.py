import numpy as np
import matplotlib.pyplot as plt
data_path = 'D:/GraduationThesis/codes/preData/part_data'
sst_X_name = 'sst_X.npy'
sst_Y_name = 'sst_Y.npy'
ssh_X_name = 'ssh_X.npy'
ssh_Y_name = 'ssh_Y.npy'
print("load data...")
sst_X = np.load(data_path + '/' + sst_X_name)
sst_Y = np.load(data_path + '/' + sst_Y_name)
ssh_X = np.load(data_path + '/' + ssh_X_name)
ssh_Y = np.load(data_path + '/' + ssh_Y_name)
print("sst_X",sst_X.shape)
print("sst_Y",sst_Y.shape)
print("ssh_X",ssh_X.shape)
print("ssh_Y",ssh_Y.shape)
print("data load finished!")
print(sst_X[0][0])
pic = sst_X[0][0]
pic = pic.transpose(1,2,0)
plt.imshow(pic)
plt.show()
