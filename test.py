### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_PRSNet
import torch

import scipy.io as sio
import os
import numpy as np
opt = TestOptions().parse(save=False)
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.noshuffle = True  # no shuffle

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
save_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# test
PRSNet = create_PRSNet(opt)
if opt.data_type == 16:
    PRSNet.half()
elif opt.data_type == 8:
    PRSNet.type(torch.uint8)

for i, data in enumerate(dataset):

    plane, quat = PRSNet.inference(data['voxel'])

    data_path = data['path'][0]
    print('[%s] process mat ... %s' % (str(i),data_path))
    matdata = sio.loadmat(data_path,verify_compressed_data_integrity=False)

    import ntpath
    short_path = ntpath.basename(data_path)
    name = os.path.splitext(short_path)[0]


    model = {'name':name, 'voxel':matdata['Volume'], 'vertices':matdata['vertices'], 'faces':matdata['faces'], 'sample':np.transpose(matdata['surfaceSamples'])}
    for j in range(opt.num_plane):
        model['plane'+str(j)] = plane[j].cpu().numpy()
    for j in range(opt.num_quat):
        model['quat'+str(j)] = quat[j].cpu().numpy()


    sio.savemat(save_dir+"/"+name+".mat",model)
