'''
load hand point data
author: Liuhao Ge
'''
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import scipy.io as sio
import pdb
from tqdm import tqdm

SAMPLE_NUM = 1024
JOINT_NUM = 16

class HandPointDataset(data.Dataset):
    def __init__(self, root_path, opt, sample=1024, train=True, shuffle=False):
        self.root_path = root_path
        self.train = train

        self.SAMPLE_NUM = sample
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM
        self.JOINT_NUM = opt.JOINT_NUM

        self.record_file, self.record_data = self.__fileToNumpy(os.path.join(root_path, 'record.txt'))

        self.total_frame_num = len(self.record_file)
        print(self.total_frame_num)
        
        self.point_clouds = np.empty(shape=[self.total_frame_num, self.SAMPLE_NUM, self.INPUT_FEATURE_NUM],
                                     dtype=np.float32)
        self.volume_length = np.empty(shape=[self.total_frame_num, 1], dtype=np.float32)
        self.gt_xyz = np.empty(shape=[self.total_frame_num, self.JOINT_NUM*3], dtype=np.float32)

        self.start_index = 0
        self.end_index = 0
        
        print("Loading Dataset..........")

        for i in tqdm(range(self.total_frame_num)):
            cur_data_dir = os.path.join(self.root_path, self.record_file[i] + '_Point_Cloud_FPS.mat')
            # print("Training: " + cur_data_dir)
            self.__loaddata(cur_data_dir)


        self.gt_xyz = self.record_data[:, 1:49].astype(np.float32)
        self.volume_length = self.record_data[:, 0].astype(np.float32)
        
        if shuffle:
            idx_shuffle = np.random.permutation(len(self.point_clouds))
            self.point_clouds = self.point_clouds[idx_shuffle]
            self.volume_length = self.volume_length[idx_shuffle]
            self.gt_xyz = self.gt_xyz[idx_shuffle]

        self.point_clouds = torch.from_numpy(self.point_clouds)
        self.volume_length = torch.from_numpy(self.volume_length).view(self.total_frame_num, 1)
        self.gt_xyz = torch.from_numpy(self.gt_xyz)

        self.total_frame_num = self.point_clouds.size(0)


    def __getitem__(self, index):
        return self.point_clouds[index, :, :], self.volume_length[index],self.gt_xyz[index, :]

    def __len__(self):
        return self.point_clouds.size(0)

    def __loaddata(self, data_dir):
        point_cloud = sio.loadmat(data_dir)

        self.start_index = self.end_index + 1
        self.end_index = self.end_index + 1

        self.point_clouds[(self.start_index - 1):self.end_index, :, :] = point_cloud['Point_Cloud_FPS'].astype(np.float32)


    def __get_frmae_num(self, data_dir):
        volume_length = sio.loadmat(os.path.join(data_dir, "Volume_length.mat"))
        return len(volume_length['Volume_length'])

    def __fileToNumpy(self, filename):
        file = open(filename)
        file_lines = file.readlines()
        numberOfLines = len(file_lines)
        dataArray = np.zeros((numberOfLines, 61))
        labels = []
        index = 0
        for line in file_lines:
            line = line.strip() # 参数为空时，默认删除开头、结尾处空白符（包括'\n', '\r',  '\t',  ' ')
            formLine = line.split(' ')
            dataArray[index,:] = formLine[1:]
            labels.append((formLine[0]))
            index += 1
        return labels, dataArray