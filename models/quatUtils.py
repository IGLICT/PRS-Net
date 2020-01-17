import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
inds = torch.LongTensor([0, -1, -2, -3, 1, 0, 3, -2, 2, -3, 0, 1, 3, 2, -1, 0]).view(4, 4)

def hamilton_product(q1, q2):
  q_size = q1.size()
  # q1 = q1.view(-1, 4)
  # q2 = q2.view(-1, 4)
  q1_q2_prods = []
  for i in range(4):
    q2_permute_0 = q2[:, :, np.abs(inds[i][0])]
    q2_permute_0 = q2_permute_0 * np.sign(inds[i][0] + 0.01)

    q2_permute_1 = q2[:, :, np.abs(inds[i][1])]
    q2_permute_1 = q2_permute_1 * np.sign(inds[i][1] + 0.01)

    q2_permute_2 = q2[:, :, np.abs(inds[i][2])]
    q2_permute_2 = q2_permute_2 * np.sign(inds[i][2] + 0.01)

    q2_permute_3 = q2[:, :, np.abs(inds[i][3])]
    q2_permute_3 = q2_permute_3 * np.sign(inds[i][3] + 0.01)
    q2_permute = torch.stack([q2_permute_0, q2_permute_1, q2_permute_2, q2_permute_3], dim=2)

    q1q2_v1 = torch.sum(q1 * q2_permute, dim=2, keepdim=True)
    q1_q2_prods.append(q1q2_v1)
  # print(q1_q2_prods[0].shape)
  q_ham = torch.cat(q1_q2_prods, dim=2)
  # q_ham = q_ham.view(q_size)
  return q_ham


def quat_conjugate(quat):
  # quat = quat.view(-1, 4)

  q0 = quat[:, :, 0]
  q1 = -1 * quat[:, :, 1]
  q2 = -1 * quat[:, :, 2]
  q3 = -1 * quat[:, :, 3]

  q_conj = torch.stack([q0, q1, q2, q3], dim=2)
  return q_conj


def quat_rot_module(points, quats):
  quatConjugate = quat_conjugate(quats)
  mult = hamilton_product(quats, points)
  mult = hamilton_product(mult, quatConjugate)
  return mult[:, :, 1:4]
