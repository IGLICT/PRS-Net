### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import scipy.io as sio
import torch
from random import randint


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_train = os.path.join(self.root, opt.phase)
        self.train_paths = sorted(make_dataset(self.dir_train))
        self.dataset_size = len(self.train_paths)

        
    def __getitem__(self, index):

        index = index % self.dataset_size
        data_path = self.train_paths[index]
        try:
            data = sio.loadmat(data_path, verify_compressed_data_integrity=False)
        except Exception as e:
            print(data_path,e)
            return None
        sample = data['surfaceSamples']
        voxel = data['Volume']
        cp = data['closestPoints']
        # vertices = data['vertices']
        # faces = data['faces']
        # axisangle = data['axisangle']

        voxel=torch.from_numpy(voxel).float().unsqueeze(0)
        sample=torch.from_numpy(sample).float().t()
        cp=torch.from_numpy(cp).float().view(-1,3)
        # print(voxel.shape,sample.shape,cp.shape)

        input_dict = {'voxel': voxel, 'sample': sample, 'cp': cp, 'path':data_path}

        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
