import numpy as np
import matplotlib.pyplot as plt


data_path = 'D:/GraduationThesis/codes/preData/all_data/ssh_X.npy'
ssh_X = np.load(data_path)
print(ssh_X.shape)
print(ssh_X[0,0,0,:,:])

plt.show()