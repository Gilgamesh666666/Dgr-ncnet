# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
# Run with python -m scripts.test_3dmatch_refactor
import os
import sys
import math
import logging
import open3d as o3d
import numpy as np
import time
import torch
import copy

sys.path.append('.')

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
# ch = logging.StreamHandler(sys.stdout)
# logging.getLogger().setLevel(logging.INFO)
# logging.basicConfig(format='%(asctime)s %(message)s',
#                     datefmt='%m/%d %H:%M:%S',
#                     handlers=[ch])

def make_open3d_point_cloud(xyz, color=None):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz)
  if color is not None:
    if len(color) != len(xyz):
      color = np.tile(color, (len(xyz), 1))
    pcd.colors = o3d.utility.Vector3dVector(color)
  return pcd

# Criteria
def rte_rre(T_pred, T_gt, rte_thresh, rre_thresh, eps=1e-16):
  if T_pred is None:
    return np.array([0, np.inf, np.inf])

  rte = np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])
  rre = np.arccos(
      np.clip((np.trace(T_pred[:3, :3].T @ T_gt[:3, :3]) - 1) / 2, -1 + eps,
              1 - eps)) * 180 / math.pi
  return np.array([rte < rte_thresh and rre < rre_thresh, rte, rre])

def analyze_stats(stats, mask, method_names):
  mask = (mask > 0).squeeze(1)
  stats = stats[:, mask, :]
  return stats.mean(0)

def create_pcd(xyz, color):
  # n x 3
  n = xyz.shape[0]
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz)
  pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (n, 1)))
  pcd.estimate_normals(
      search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
  return pcd

class ThreeDMatchTrajectoryDataset_info(torch.utils.data.Dataset):
  '''
  Test dataset
  '''
  DATA_FILES = {
      'train': './dataloader/split/train_3dmatch.txt',
      'val': './dataloader/split/val_3dmatch.txt',
      'test': './dataloader/split/test_3dmatch.txt'
  }

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               scene_id=None,
               config=None,
               return_ply_names=False):
    self.return_ply_names = return_ply_names
    self.phase = phase
    self.files = []
    self.data_objects = []
    self.transform = transform
    self.voxel_size = config.voxel_size
    self.matching_search_voxel_size = \
        config.voxel_size * config.positive_pair_search_voxel_size_multiplier

    self.random_scale = random_scale
    self.min_scale = config.min_scale
    self.max_scale = config.max_scale
    self.random_rotation = random_rotation
    self.rotation_range = config.rotation_range
    self.randg = np.random.RandomState()
    if manual_seed:
      self.reset_seed()
    self.root = config.threed_match_dir

    subset_names = open(self.DATA_FILES[phase]).read().split()
    if scene_id is not None:
      subset_names = [subset_names[scene_id]]
    for sname in subset_names:
      log_filedir = os.path.join('/home/zebai/test_all/3DMatch/gt_info/3dmatch', sname + '-evaluation')
      traj = read_trajectory_info(log_filedir)
      for ctraj in traj:
        i = ctraj.metadata[0]
        j = ctraj.metadata[1]
        T_gt = ctraj.pose
        T_info = ctraj.info
        self.files.append((sname, i, j, T_gt, T_info))

  def reset_seed(self, seed=0):
    logging.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts
  def __len__(self):
    return len(self.files)
  def __getitem__(self, pair_index):
    sname, i, j, T_gt, T_info = self.files[pair_index]
    ply_name0 = os.path.join(self.root, sname, f'cloud_bin_{i}.ply')
    ply_name1 = os.path.join(self.root, sname, f'cloud_bin_{j}.ply')

    if self.return_ply_names:
      return sname, ply_name0, ply_name1, T_gt, T_info

    pcd0 = o3d.io.read_point_cloud(ply_name0)
    pcd1 = o3d.io.read_point_cloud(ply_name1)
    pcd0 = np.asarray(pcd0.points)
    pcd1 = np.asarray(pcd1.points)
    save_subpath = f'cloud_bin_{i}_{j}'
    return save_subpath, sname, pcd0, pcd1, T_gt, T_info

def read_trajectory_info(filedir):
  
  class CameraPose_Info:
    def __init__(self, meta, mat, info):
      self.metadata = meta
      self.pose = mat
      self.info = info
    def __str__(self):
      return 'metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
        "pose : " + "\n" + np.array_str(self.pose) + \
        "info : " + "\n" + np.array_str(self.info)
  logfile = os.path.join(filedir, 'gt.log')
  infofile = os.path.join(filedir, 'gt.info')
  assert os.path.exists(logfile)
  assert os.path.exists(infofile)
  with open(logfile, 'r') as f:
      gtlog = f.readlines()
  with open(infofile, 'r') as f:
      gtinfo = f.readlines()
  traj = []
  i, j = 0, 0
  while i < len(gtlog) and j < len(gtinfo):
    logline = gtlog[i].strip().split('\t')
    infoline = gtinfo[j].strip().split('\t')
    assert len(logline)==3
    assert len(infoline)==3
    head = [int(s) for s in logline]
    trans = np.array([[float(s) for s in gtlog[i+1].strip().split('\t')],
                    [float(s) for s in gtlog[i+2].strip().split('\t')],
                    [float(s) for s in gtlog[i+3].strip().split('\t')],
                    [float(s) for s in gtlog[i+4].strip().split('\t')]])
    info = np.array([[float(s) for s in gtinfo[j+1].strip().split('\t')],
                    [float(s) for s in gtinfo[j+2].strip().split('\t')],
                    [float(s) for s in gtinfo[j+3].strip().split('\t')],
                    [float(s) for s in gtinfo[j+4].strip().split('\t')],
                    [float(s) for s in gtinfo[j+5].strip().split('\t')],
                    [float(s) for s in gtinfo[j+6].strip().split('\t')]])
    traj.append(CameraPose_Info(head, trans, info))
    i += 5
    j += 7
  return traj

def evaluate(model, data_loader, config, debug=False):

  tot_num_data = len(data_loader.dataset)
  data_loader_iter = iter(data_loader)
  # Accumulate success, rre, rte, time, sid
  mask = np.zeros((tot_num_data, 1)).astype(int)
  stats = np.zeros((tot_num_data, 5))

  dataset = data_loader.dataset
  subset_names = open(dataset.DATA_FILES[dataset.phase]).read().split()

  for batch_idx in range(tot_num_data):
    batch = data_loader_iter.next()

    # Skip too sparse point clouds
    save_subpath, sname, xyz0, xyz1, trans, info = batch[0]

    pred_trans_save_dir = os.path.join(config.test_save_dir, sname)
    os.makedirs(pred_trans_save_dir, exist_ok=True)

    sid = subset_names.index(sname)
    T_gt = np.linalg.inv(trans)

    start = time.time()
    T = model.register(xyz0, xyz1)
    reg_time = time.time() - start

    np.savez(os.path.join(pred_trans_save_dir, save_subpath+'.npz'), trans=T,time=reg_time)
    logging.info(f'{save_subpath} Saved')
    # stats[batch_idx, :3] = rte_rre(T, T_gt, config.success_rte_thresh,
    #                                   config.success_rre_thresh)
    # stats[batch_idx, 3] = end - start
    # stats[batch_idx, 4] = sid
    # mask[batch_idx] = 1
    # if stats[batch_idx, 0] == 0: # otherwise is True or False
    #   print(f"{method_names[i]}: failed")
    #print(stats)
  # # Save results
  # filename = f'3dmatch-stats_{method.__class__.__name__}'
  # if os.path.isdir(config.out_dir):
  #   out_file = os.path.join(config.out_dir, filename)
  # else:
  #   out_file = filename  # save it on the current directory
  # print(f'Saving the stats to {out_file}')
  # np.save(out_file, stats)
  
  # stats_mean = analyze_stats(stats, mask, method_names)

  # # Analysis per scene
  # for i, method in enumerate(methods):
  #   print(f'Scene-wise mean {method}')
  #   scene_vals = np.zeros((len(subset_names), 3))
  #   for sid, sname in enumerate(subset_names):
  #     curr_scene = stats[i, :, 4] == sid
  #     scene_vals[sid] = (stats[i, curr_scene, :3]).mean(0)

  #   print('All scenes')
  #   print(scene_vals)
  #   print('Scene average')
  #   print(scene_vals.mean(0))


if __name__ == '__main__':
  from config import get_config
  from core.deep_global_registration import DeepGlobalRegistration
  from datetime import datetime
  now = datetime.now()
  
  config = get_config()
  logging.basicConfig(filename=f'test_3dmatch_{now.strftime("%m_%d_%H_%M")}.log',format='%(asctime)s %(message)s',
                      datefmt='%m/%d %H:%M:%S',level=logging.INFO)#,
  logging.info(config)
  
  dgr = DeepGlobalRegistration(config)


  dset = ThreeDMatchTrajectoryDataset_info(phase='test',
                                      transform=None,
                                      random_scale=False,
                                      random_rotation=False,
                                      config=config)

  data_loader = torch.utils.data.DataLoader(dset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1,
                                            collate_fn=lambda x: x,
                                            pin_memory=False,
                                            drop_last=True)

  evaluate(dgr, data_loader, config, debug=False)
