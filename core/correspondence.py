# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import copy
import numpy as np

import open3d as o3d
import torch
from util.hash import _hash



def find_correct_correspondence(pos_pairs, pred_pairs, hash_seed=None, len_batch=None):
  assert len(pos_pairs) == len(pred_pairs)
  if hash_seed is None:
    assert len(len_batch) == len(pos_pairs)

  corrects = []
  for i, pos_pred in enumerate(zip(pos_pairs, pred_pairs)):
    pos_pair, pred_pair = pos_pred
    if isinstance(pos_pair, torch.Tensor):
      pos_pair = pos_pair.numpy()
    if isinstance(pred_pair, torch.Tensor):
      pred_pair = pred_pair.numpy()

    if hash_seed is None:
      N0, N1 = len_batch[i]
      _hash_seed = max(N0, N1)
    else:
      _hash_seed = hash_seed

    pos_keys = _hash(pos_pair, _hash_seed)
    pred_keys = _hash(pred_pair, _hash_seed)

    corrects.append(np.isin(pred_keys, pos_keys, assume_unique=False))

  return np.hstack(corrects)

