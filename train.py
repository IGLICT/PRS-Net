from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_PRSNet
from utils.visualizer import Visualizer
import numpy as np
import torch
import os
import time
from torchsummary import summary


torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
opt = TrainOptions().parse()

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

PRSNet = create_PRSNet(opt)
print(PRSNet)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq
visualizer = Visualizer(opt)


for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()

    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    for i, data in enumerate(dataset):
        
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize


        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        losses = PRSNet(data['voxel'], data['sample'], data['cp'])
        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        losses_dict = dict(zip(PRSNet.loss_names, losses))

        loss = (losses_dict['ref'] + losses_dict['rot'] + losses_dict['reg_plane'] + losses_dict['reg_rot'])

        ############### Backward Pass ####################
        # update generator weights
        PRSNet.optimizer_PRS.zero_grad()
        loss.backward()
        PRSNet.optimizer_PRS.step()

        ############## Display results and errors ##########
        ### print out errors
        print(total_steps, opt.print_freq, print_delta)
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in losses_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            visualizer.plot_current_weights(PRSNet, total_steps)
            visualizer.print_line('')

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            PRSNet.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        PRSNet.save('latest')
        PRSNet.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
