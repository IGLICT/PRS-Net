import torch
import torch.nn as nn
from .quatUtils import  quat_conjugate, quat_rot_module

def rigidTsdf(points, trans, quat):
  p1 = translate_module(points, -1*trans)
  p2 = rotate_module(p1, quat)
  # -- performs p_out = R'*p_in + t
  return p2


def planesymTransform(sample, plane):
  abc = plane[:,0:3].unsqueeze(1).repeat(1,sample.shape[1],1)
  d = plane[:,3].unsqueeze(1).unsqueeze(1).repeat(1,sample.shape[1],1)
  fenzi = torch.sum(sample*abc,2,True)+d
  fenmu = torch.norm(plane[:,0:3],2,1,True).unsqueeze(1).repeat(1,sample.shape[1],1)+1e-5
  x = 2*torch.div(fenzi,fenmu)
  y=torch.mul(x.repeat(1,1,3),abc/fenmu)
  return sample-y

def rotsymTransform(sample, quat):
  return rotate_module(sample, quat)


def rigidPointsTransform(points, trans, quat):
  quatConj = quat_conjugate(quat)
  p1 = rotate_module(points, quatConj)
  p2 = translate_module(p1, trans)
  return p2

## points is Batch_size x P x 3,  #Bx4 quat vectors
def rotate_module(points, quat):
  nP = points.size(1)
  quat_rep = quat.unsqueeze(1).repeat(1, nP, 1)
  # print(quat_rep.shape)
  zero_points = 0 * points[:, :, 0].clone().view(-1, nP, 1)
  quat_points = torch.cat([zero_points, points], dim=2)

  rotated_points = quat_rot_module(quat_points, quat_rep)  # B x  P x 3
  return rotated_points

def translate_module(points, trans):
  nP = points.size(1)
  trans_rep = trans.repeat(1, nP, 1)

  return points + trans_rep