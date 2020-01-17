from .base_model import BaseModel
from .network import *
import numpy as np
from torch.autograd import Variable
class PRSNet(BaseModel):
    def name(self):
        return 'PRSNet'
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        biasTerms={}
        biasTerms['plane1']=[1,0,0,0]
        biasTerms['plane2']=[0,1,0,0]
        biasTerms['plane3']=[0,0,1,0]
        biasTerms['quat1']=[0, 0, 0, np.sin(np.pi/2)]
        biasTerms['quat2']=[0, 0, np.sin(np.pi/2), 0]
        biasTerms['quat3']=[0, np.sin(np.pi/2), 0, 0]
        if opt.num_plane > 3:
            for i in range(4,opt.num_plane+1):
                plane = np.random.random_sample((3,))
                biasTerms['plane'+str(i)] = (plane/np.linalg.norm(plane)).tolist()+[0]
        if opt.num_quat > 3:
            for i in range(4,opt.num_quat+1):
                quat = np.random.random_sample((4,))
                biasTerms['quat'+str(i)] = (quat/np.linalg.norm(quat)).tolist()

        self.opt = opt
        self.netPRS = define_PRSNet(opt.input_nc, opt.ngf, opt.conv_layers, opt.num_plane, opt.num_quat, biasTerms,gpu_ids=self.gpu_ids)


        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netPRS, self.name() + 'PRS', opt.which_epoch, pretrained_path)
        if self.isTrain:
            self.sym_loss = symLoss(opt.gridBound, opt.gridSize)
            self.reg_loss = RegularLoss()
            self.loss_names = ['ref', 'rot', 'reg_plane', 'reg_rot']
            params = list(self.netPRS.parameters())
            self.optimizer_PRS = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))


    def forward(self, voxel, points, cp):

        voxel = Variable(voxel.data.cuda(), requires_grad=True)
        points = Variable(points.data.cuda())
        cp = Variable(cp.data.cuda())
        quat, plane = self.netPRS(voxel)
        # print(quat,plane)
        loss_ref, loss_rot = self.sym_loss(points, cp, voxel, plane = plane, quat = quat)
        loss_reg_plane, loss_reg_rot = self.reg_loss(plane = plane, quat = quat, weight=self.opt.weights)

        return [loss_ref, loss_rot,loss_reg_plane, loss_reg_rot]

    def inference(self, voxel):
        if len(self.gpu_ids) > 0:
            voxel = Variable(voxel.data.cuda())
        else:
            voxel = Variable(voxel.data)
        self.netPRS.eval()
        with torch.no_grad():
            quat, plane = self.netPRS(voxel)
        return plane, quat

    def save(self, which_epoch):
        self.save_network(self.netPRS, self.name() + 'PRS', which_epoch, self.gpu_ids)



class Inference(PRSNet):
    def forward(self, voxel):
        return self.inference(voxel)