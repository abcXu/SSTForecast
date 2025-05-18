import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
# data_path = 'D:/GraduationThesis/codes/preData/part_data'
# data_path = './data/all_data'
class SCSDataset(Dataset):
    def __init__(self,args,split):
        self.args = args
        self.split = split
        self.data_path = args.data_path
        # self.data_path = data_path
        self.sst_X_name = 'sst_X.npy'
        self.sst_Y_name = 'sst_Y.npy'
        self.ssh_X_name = 'ssh_X.npy'
        self.ssh_Y_name = 'ssh_Y.npy'
        self.sst_X = np.load(self.data_path + '/' + self.sst_X_name)
        self.sst_Y = np.load(self.data_path + '/' + self.sst_Y_name)
        self.ssh_X = np.load(self.data_path + '/' + self.ssh_X_name)
        self.ssh_Y = np.load(self.data_path + '/' + self.ssh_Y_name)

        # train : valid : test = 7 : 1 : 2
        if self.split == 'train':
            self.sst_X = self.sst_X[:int(len(self.sst_X)*0.7)]
            self.sst_Y = self.sst_Y[:int(len(self.sst_Y)*0.7)]
            self.ssh_X = self.ssh_X[:int(len(self.ssh_X)*0.7)]
            self.ssh_Y = self.ssh_Y[:int(len(self.ssh_Y)*0.7)]

        if self.split == 'valid':
            self.sst_X = self.sst_X[int(len(self.sst_X)*0.7):int(len(self.sst_X)*0.8)]
            self.sst_Y = self.sst_Y[int(len(self.sst_Y)*0.7):int(len(self.sst_Y)*0.8)]
            self.ssh_X = self.ssh_X[int(len(self.ssh_X)*0.7):int(len(self.ssh_X)*0.8)]
            self.ssh_Y = self.ssh_Y[int(len(self.ssh_Y)*0.7):int(len(self.ssh_Y)*0.8)]

        if self.split == 'test':
            self.sst_X = self.sst_X[int(len(self.sst_X)*0.8):]
            self.sst_Y = self.sst_Y[int(len(self.sst_Y)*0.8):]
            self.ssh_X = self.ssh_X[int(len(self.ssh_X)*0.8):]
            self.ssh_Y = self.ssh_Y[int(len(self.ssh_Y)*0.8):]
        print("the length of {} set :".format(self.split))
        print("sst_X",self.sst_X.shape)
        print("sst_Y",self.sst_Y.shape)
        print("ssh_X",self.ssh_X.shape)
        print("ssh_Y",self.ssh_Y.shape)
        print("data load finished!")

    def __len__(self):
        return len(self.sst_X)
    def __getitem__(self, index):
        input_sst_x = torch.from_numpy(self.sst_X[index])
        target_sst_y = torch.from_numpy(self.sst_Y[index])
        input_ssh_x = torch.from_numpy(self.ssh_X[index])
        target_ssh_y = torch.from_numpy(self.ssh_Y[index])
        return input_sst_x,target_sst_y,input_ssh_x,target_ssh_y


if __name__ == '__main__':
    args = {
        'data_path':'./data/all_data',
        'sst_X_name':'sst_X.npy',
        'sst_Y_name':'sst_Y.npy',
        'ssh_X_name':'ssh_X.npy',
        'ssh_Y_name':'ssh_Y.npy',
        'split':'train'
    }
    # train_set = SCSDataset(args,args['split'])
    test_set = SCSDataset(args,'test')
    # valid_set = SCSDataset(args,'valid')
