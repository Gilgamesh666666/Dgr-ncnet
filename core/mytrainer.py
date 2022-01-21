# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License
import time
import os
import os.path as osp
import gc
import logging
import numpy as np
import json

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from model import load_model
from core.knn import find_knn_batch
from core.correspondence import find_correct_correspondence
from core.loss import UnbalancedLoss, BalancedLoss
from core.metrics import batch_rotation_error, batch_translation_error
import core.registration as GlobalRegistration

from util.timer import Timer, AverageMeter
from util.file import ensure_dir
from util.hash import _hash
import copy
import MinkowskiEngine as ME
import traceback
from extension.sms import soft_mutual_score_Module
import numba as nb
eps = np.finfo(float).eps
np2th = torch.from_numpy

#@nb.jit(nopython=True)
def checknan(tensor): 
  if isinstance(tensor, np.ndarray): 
    return np.isnan(tensor).astype(np.int16).sum() 
  elif isinstance(tensor, torch.Tensor): 
    return torch.isnan(tensor).int().sum().item() 
  elif isinstance(tensor, (list, tuple)):
    for item in tensor:
      if checknan(item):
        return True
  else:
    return NotImplemented
#@nb.jit(nopython=True)
def checkinf(tensor): 
  if isinstance(tensor, np.ndarray): 
    return np.isinf(tensor).astype(np.int16).sum() 
  elif isinstance(tensor, torch.Tensor): 
    return torch.isinf(tensor).int().sum().item() 
  elif isinstance(tensor, (list, tuple)):
    for item in tensor:
      if checkinf(item):
        return True
  else:
    return NotImplemented

def CHECK(tensor, name):
  nannum = checknan(tensor)
  infnum = checkinf(tensor)
  if  nannum or infnum:
    if not nannum:
      logging.info(f'{name} has {infnum} inf\n{name}={tensor}')
      exit(-1)
    elif not infnum: 
      logging.info(f'{name} has {nannum} nan\n{name}={tensor}')
      exit(-1)
    else:
      logging.info(f'{name} has {nannum} nan and {infnum} inf\n{name}={tensor}')
      exit(-1)

def batch_CHECK(tensors, names):
  for tensor, name in zip(tensors, names):
    CHECK(tensor, name)

class WeightedProcrustesTrainer:
  def __init__(self, config, data_loader, val_data_loader=None):
    # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.
    num_feats = 3 if config.use_xyz_feature else 1

    # Feature model initialization
    if config.use_gpu and not torch.cuda.is_available():
      logging.warning('Warning: There\'s no CUDA support on this machine, '
                      'training is performed on CPU.')
      raise ValueError('GPU not available, but cuda flag set')
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.config = config

    # Training config
    self.max_epoch = config.max_epoch
    self.start_epoch = 1
    self.checkpoint_dir = config.out_dir

    self.data_loader = data_loader
    self.train_data_loader_iter = self.data_loader.__iter__()

    self.iter_size = config.iter_size
    self.batch_size = data_loader.batch_size
    self.ws_thresh = config.ws_thresh
    # Validation config
    self.val_max_iter = config.val_max_iter
    self.val_epoch_freq = config.val_epoch_freq
    self.best_val_metric = config.best_val_metric
    self.best_val_epoch = -np.inf
    self.best_val = -np.inf

    self.with_sms = config.with_sms

    self.val_data_loader = val_data_loader
    self.test_valid = True if self.val_data_loader is not None else False

    # Logging
    self.log_step = int(np.sqrt(self.config.batch_size))
    self.writer = SummaryWriter(config.out_dir)

    # Model
    FeatModel = load_model(config.feat_model)
    InlierModel = load_model(config.inlier_model)

    num_feats = 6 if self.config.inlier_feature_type == 'coords' else 1
    self.feat_model = FeatModel(num_feats,
                                config.feat_model_n_out,
                                bn_momentum=config.bn_momentum,
                                conv1_kernel_size=config.feat_conv1_kernel_size,
                                normalize_feature=config.normalize_feature).to(
                                    self.device)
    #logging.info(self.feat_model)

    self.inlier_model = InlierModel(num_feats,
                                    1,
                                    bn_momentum=config.bn_momentum,
                                    conv1_kernel_size=config.inlier_conv1_kernel_size,
                                    normalize_feature=False,
                                    D=6).to(self.device)
    #logging.info(self.inlier_model)

    # Loss and optimizer
    self.clip_weight_thresh = self.config.clip_weight_thresh
    if self.config.use_balanced_loss:
      self.crit = BalancedLoss()
    else:
      self.crit = UnbalancedLoss()

    self.optimizer = getattr(optim, config.optimizer)(self.inlier_model.parameters(),
                                                      lr=config.lr,
                                                      momentum=config.momentum,
                                                      weight_decay=config.weight_decay)
    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)

    # Output preparation
    ensure_dir(self.checkpoint_dir)
    json.dump(config,
              open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
              indent=4,
              sort_keys=False)

    self._load_weights(config)


  def train(self):
    """
    Major interface
    Full training logic: train, valid, and save
    """
    # Baseline random feature performance
    if self.test_valid:
      val_dict = self._valid_epoch()
      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, 0)

    # Train and valid
    for epoch in range(self.start_epoch, self.max_epoch + 1):
      lr = self.scheduler.get_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch(epoch)
      self._save_checkpoint(epoch)
      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        val_dict = self._valid_epoch()
        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)

        if self.best_val < val_dict[self.best_val_metric]:
          logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
          )
          self.best_val = val_dict[self.best_val_metric]
          self.best_val_epoch = epoch
          self._save_checkpoint(epoch, 'best_val_checkpoint')

        else:
          logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
          )
  def inference(self, input_dict):
    # unpack input dictionary
    xyz0s=input_dict['pcd0']
    xyz1s=input_dict['pcd1']
    iC0=input_dict['sinput0_C']
    iC1=input_dict['sinput1_C']
    iF0=input_dict['sinput0_F']
    iF1=input_dict['sinput1_F']
    len_batch=input_dict['len_batch']
    pos_pairs=input_dict['correspondences']
    batch_CHECK((xyz0s, xyz1s, iC0, iC1, iF0, iF1, len_batch, pos_pairs), ('xyz0s','xyz1s','iC0','iC1','iF0', 'iF1', 'len_batch', 'pos_pairs'))
    
    # for xyz0, xyz1, lens in zip(xyz0s, xyz1s, len_batch):
    #     print(f'xyz0={xyz0.shape}, xyz1={xyz1.shape}, lens={lens}')
    stime = time.time()
    sinput0 = ME.SparseTensor(iF0, coords=iC0).to(self.device)
    oF0 = self.feat_model(sinput0).F

    sinput1 = ME.SparseTensor(iF1, coords=iC1).to(self.device)
    oF1 = self.feat_model(sinput1).F
    feat_time = time.time() - stime
    batch_CHECK((oF0, oF1), ('oF0', 'oF1'))
    
    # TODO:
    # oF0 find knn in oF1
    # oF1 find knn in oF0
    # concatenate them
    # delete dulplicate
    # calculate the correlation between knns
    # add soft mutual score
    # consensus fliter
    # add soft mutual score
    # bilateral matches supervise
    
    stime = time.time()
    # with torch.no_grad():
    #   total_matches, bil_agree_matches = self.find_pairs_bilateral(oF0, oF1, len_batch, self.config.inlier_knn, False)
    #---------------------------------------------------------
    len_batch_reverse = copy.deepcopy(len_batch)
    len_batch_reverse[:, [0, 1]] = len_batch_reverse[:, [1, 0]]
    
    with torch.no_grad():
      # oF0 find knn in oF1
      pred_matches01 = self.find_pairs(xyz0s, xyz1s, oF0, oF1, len_batch, self.config.inlier_knn, self.config.nn_max_n, False, self.config.knn_search_method)
      # oF1 find knn in oF0
      reverse_pred_matches10 = self.find_pairs(xyz1s, xyz0s, oF1, oF0, len_batch_reverse, self.config.inlier_knn, self.config.nn_max_n, False, self.config.knn_search_method)
      pred_matches10 = []
      for nns10 in reverse_pred_matches10:
        nns10[:, [0,1]] = nns10[:, [1,0]]
        pred_matches10.append(nns10)
      # concatenate them, delete dulplicate
      total_matches = []
      bil_agree_matches = []
      for xyz0, xyz1, nns01, nns10, lens in zip(xyz0s, xyz1s, pred_matches01, pred_matches10, len_batch):
        N0, N1 = lens
        hashseed = max(N0, N1)
        hash_vec01 = _hash(nns01.cpu().numpy(), hashseed)
        hash_vec10 = _hash(nns10.cpu().numpy(), hashseed)
        mask = np.isin(hash_vec10, hash_vec01)
        mask_not = np.logical_not(mask)
        totalnns = torch.cat((nns01, nns10[mask_not, :]), 0)
        total_matches.append(totalnns)
        bil_agree_matches.append(nns10[mask, :])
    #-----------------------------------------------------------------
    nn_time = time.time() - stime
    batch_CHECK((total_matches, bil_agree_matches), ('total_matches', 'bil_agree_matches'))
    stime = time.time()
    # calculate the correlation between knns
    with torch.no_grad():
      cat_pred_pairs = []
      start_inds = torch.zeros((1, 2)).long()
      for lens, pred_pairs in zip(len_batch, total_matches):
        cat_pred_pairs.append(pred_pairs + start_inds)
        start_inds += torch.LongTensor(lens)
      cat_pred_pairs = torch.cat(cat_pred_pairs, 0)
    feat0 = F.normalize(oF0[cat_pred_pairs[:, 0]], dim=1) # [n1+n2+..nb, c]
    feat1 = F.normalize(oF1[cat_pred_pairs[:, 1]], dim=1) # [n1+n2+..nb, c]

    cat_reg_feat_batch = torch.sum((feat0*feat1), dim=1, keepdim=True) # [n1+n2+..nb, 1]
    CHECK(cat_reg_feat_batch, 'cat_reg_feat_batch')
    # add soft mutual score
    reg_feat_batch = self.decompose_by_length(cat_reg_feat_batch, total_matches)
    cat_reg_feat_batch = []
    soft_mutual_cache = []
    sms = soft_mutual_score_Module()
    for pred_pair, reg_feat in zip(total_matches, reg_feat_batch):
      batch_CHECK((pred_pair, reg_feat), ('pred_pair', 'reg_feat'))
      if self.with_sms:
        soft_mutual_score = sms(pred_pair.cuda(), reg_feat)
        CHECK(soft_mutual_score, 'soft_mutual_score')
        reg_feat = reg_feat.pow(3)/soft_mutual_score
      cat_reg_feat_batch.append(reg_feat)
      #soft_mutual_cache.append((len(output0), inverse_indices0, len(output1), inverse_indices1))

    cat_reg_feat_batch = torch.cat(cat_reg_feat_batch, 0)
    # consensus fliter N(Cab)+N(Cab.t).t
    reg_coords = torch.cat((iC0[cat_pred_pairs[:, 0]], iC1[cat_pred_pairs[:, 1], 1:]), dim=1)
    reg_sinput = ME.SparseTensor(cat_reg_feat_batch.contiguous(), coords=reg_coords.int()).to(self.device)
    batch_CHECK((reg_sinput.coords, reg_sinput.F), ('reg_sinput.coords', 'reg_sinput.feat'))
    corr6d = self.inlier_model(reg_sinput)
    batch_CHECK((corr6d.coords, corr6d.F), ('corr6d.coords', 'corr6d.feat'))
    
    reg_coords_reverse = torch.cat((iC1[cat_pred_pairs[:, 1]], iC0[cat_pred_pairs[:, 0], 1:]), dim=1)
    reg_sinput_reverse = ME.SparseTensor(cat_reg_feat_batch.contiguous(), coords=reg_coords_reverse.int()).to(self.device)
    # print(reg_sinput_reverse.coords)
    # print(cat_reg_feat_batch)
    # print(reg_sinput_reverse.F)
    corr6d_temp = self.inlier_model(reg_sinput_reverse)
    batch_CHECK((corr6d_temp.coords, corr6d_temp.F), ('corr6d_temp.coords', 'corr6d_temp.feat'))
    
    corr6d_reverse = ME.SparseTensor(corr6d_temp.feats.clone(), coords=corr6d_temp.coords[:, [0,4,5,6,1,2,3]].clone(), coords_manager=corr6d.coords_man,  force_creation=True)
    scorr = ME.MinkowskiUnion()(corr6d,corr6d_reverse)
    CHECK(scorr.coords, 'scorr.coords')
    inlier_time = time.time() - stime
    # # add soft mutual score when test
    # if is_test:
    #   corr6d_score = corr6d.F
    #   for i in range(output0):
    #       max_c_ijab = reg_feat[inverse_indices0==i].max().values.item()
    #       soft_mutual_score[inverse_indices0==i] *= max_c_ijab
    #   for i in range(output1):
    #     max_c_cdkl = reg_feat[inverse_indices1==i].max().values.item()
    #     soft_mutual_score[inverse_indices1==i] *= max_c_cdkl

    # bilateral matches supervise

    stime = time.time()
    logits = scorr.F
    weights = logits.sigmoid()
    batch_CHECK((logits, weights), ('logits', 'weights'))
    
    # Truncate weights too low
    # For training, inplace modification is prohibited for backward
    if self.clip_weight_thresh > 0:
      weights_tmp = torch.zeros_like(weights)
      valid_mask = weights > self.clip_weight_thresh
      weights_tmp[valid_mask] = weights[valid_mask]
      weights = weights_tmp
      # Weighted Procrustes
    pred_rots, pred_trans, ws = self.weighted_procrustes(xyz0s=xyz0s,
                                                          xyz1s=xyz1s,
                                                          pred_pairs=total_matches,
                                                          weights=weights)
    batch_CHECK((pred_rots, pred_trans, ws), ('pred_rots', 'pred_trans', 'ws'))
    dgr_time = time.time() - stime
    return pred_rots, pred_trans, ws, weights, logits, total_matches, bil_agree_matches, (feat_time, nn_time, inlier_time, dgr_time)

  def _train_epoch(self, epoch):
    gc.collect()

    # Fix the feature model and train the inlier model
    self.feat_model.eval()
    self.inlier_model.train()

    # Epoch starts from 1
    total_loss, total_num = 0, 0.0
    data_loader = self.data_loader
    iter_size = self.iter_size

    # Meters for statistics
    average_valid_meter = AverageMeter()
    loss_meter = AverageMeter()
    data_meter = AverageMeter()
    regist_succ_meter = AverageMeter()
    regist_rte_meter = AverageMeter()
    regist_rre_meter = AverageMeter()
    hit_ratio_meter = AverageMeter()
    # Timers for profiling
    data_timer = Timer()
    nn_timer = Timer()
    inlier_timer = Timer()
    total_timer = Timer()

    if self.config.num_train_iter > 0:
      num_train_iter = self.config.num_train_iter
    else:
      num_train_iter = len(data_loader) // iter_size
    start_iter = (epoch - 1) * num_train_iter

    tp, fp, tn, fn = 0, 0, 0, 0

    # Iterate over batches
    for curr_iter in range(num_train_iter):
      self.optimizer.zero_grad()

      batch_loss, data_time = 0, 0
      total_timer.tic()

      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = self.get_data(self.train_data_loader_iter)
        data_time += data_timer.toc(average=False)
        xyz0s=input_dict['pcd0']
        xyz1s=input_dict['pcd1']
        len_batch=input_dict['len_batch']
        # for xyz0, xyz1, lens in zip(xyz0s, xyz1s, len_batch):
        #     print(f'before xyz0={xyz0.shape}, xyz1={xyz1.shape}, lens={lens}')
        # Initial inlier prediction with FCGF and KNN matching
        # 6维
        pred_rots, pred_trans, ws, weights, logits, total_matches, bil_agree_matches, time_records = self.inference(input_dict)
        # Get batch registration loss
        gt_rots, gt_trans = self.decompose_rotation_translation(input_dict['T_gt'])
        rot_error = batch_rotation_error(pred_rots, gt_rots)
        trans_error = batch_translation_error(pred_trans, gt_trans)
        individual_loss = rot_error + self.config.trans_weight * trans_error
        CHECK(individual_loss, 'individual_loss')
        # Select batches with at least 10 valid correspondences
        valid_mask = ws > self.ws_thresh
        num_valid = valid_mask.sum().item()

        # Registration loss against registration GT
        loss = self.config.procrustes_loss_weight * individual_loss[valid_mask].mean()
        reg_loss = loss.item()
        if not np.isfinite(loss.item()):
          max_val = loss.item()
          logging.info(f'individual_loss = {individual_loss}')
          logging.info(f'ws = {ws.data}')
          logging.info(f'num_valid = {num_valid}')
          logging.info(f'Loss is infinite, abort max_val={max_val}')
          continue
        # 之前算出来的is_correct作为监督,监督correspondence confidence prediction的结果
        # Direct inlier loss against nearest neighbor searched GT
        is_correct = find_correct_correspondence(input_dict['correspondences'], total_matches, len_batch=input_dict['len_batch'])
        target = torch.from_numpy(is_correct).squeeze()
        if self.config.inlier_use_direct_loss:
          inlier_loss = self.config.inlier_direct_loss_weight * self.crit(
              logits.cpu().squeeze(), target.to(torch.float)) / iter_size
          loss += inlier_loss
        current_step = iter_idx + curr_iter*iter_size + epoch*num_train_iter*iter_size
        self.writer.add_scalar(f'train/total_loss', loss.item(), current_step)
        self.writer.add_scalar(f'train/reg_loss', reg_loss, current_step)
        self.writer.add_scalar(f'train/inlier_loss', inlier_loss, current_step)
        logging.info(f'Registration loss={reg_loss}, total loss={loss.item()}, inlier_loss={inlier_loss}')
        loss.backward()

        # Update statistics before backprop
        with torch.no_grad():
          feat_time, nn_time, inlier_time, dgr_time = time_records
          average_valid_meter.update(num_valid)
          nn_timer.update(nn_time)
          inlier_timer.update(inlier_time)
          data_meter.update(data_time)
          is_correct_fot_hit_ratio = find_correct_correspondence(input_dict['correspondences'], bil_agree_matches, len_batch=input_dict['len_batch'])
          hit_ratio_meter.update(is_correct_fot_hit_ratio.sum().item() / len(is_correct_fot_hit_ratio))

          regist_rre_meter.update(rot_error.squeeze() * 180 / np.pi)
          regist_rte_meter.update(trans_error.squeeze())

          success = (trans_error.squeeze() < self.config.success_rte_thresh) * (
              rot_error.squeeze() * 180 / np.pi < self.config.success_rre_thresh)
          regist_succ_meter.update(success.float())

          batch_loss += loss.mean().item()

          neg_target = (~target).to(torch.bool)
          pred = logits > 0  # todo thresh
          pred_on_pos, pred_on_neg = pred[target], pred[neg_target]
          tp += pred_on_pos.sum().item()
          fp += pred_on_neg.sum().item()
          tn += (~pred_on_neg).sum().item()
          fn += (~pred_on_pos).sum().item()

          # Check gradient and avoid backprop of inf values
          max_grad = torch.abs(self.inlier_model.final.kernel.grad).max().cpu().item()

        # Backprop only if gradient is finite
        if not np.isfinite(max_grad):
          self.optimizer.zero_grad()
          logging.info(f'Clearing the NaN gradient at iter {curr_iter}')
        else:
          self.optimizer.step()

      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      loss_meter.update(batch_loss)

      # Output to logs
      if (curr_iter) % self.config.stat_freq == 0:
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        tpr = tp / (tp + fn + eps)
        tnr = tn / (tn + fp + eps)
        balanced_accuracy = (tpr + tnr) / 2

        logging.info(' '.join([
            f"Train Epoch: {epoch} [{curr_iter}/{num_train_iter}],",
            f"Current Loss: {loss_meter.avg:.3f},",
            f"hit ratio: {hit_ratio_meter.avg:.3f}",
            f", Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f},",
            f"TPR: {tpr:.4f}, TNR: {tnr:.4f}, BAcc: {balanced_accuracy:.4f}",
            f"RTE: {regist_rte_meter.avg:.3f}, RRE: {regist_rre_meter.avg:.3f},",
            f"Succ rate: {regist_succ_meter.avg:3f}",
            f"Avg num valid: {average_valid_meter.avg:3f}",
            f"\tData time: {data_meter.avg:.4f}, Train time: {total_timer.avg - data_meter.avg:.4f},",
            f"NN search time: {nn_timer.avg:.3f}, Total time: {total_timer.avg:.4f}"
        ]))

        loss_meter.reset()
        regist_rte_meter.reset()
        regist_rre_meter.reset()
        regist_succ_meter.reset()
        hit_ratio_meter.reset()
        average_valid_meter.reset()
        data_meter.reset()
        total_timer.reset()

        tp, fp, tn, fn = 0, 0, 0, 0

  def _valid_epoch(self):
    # Change the network to evaluation mode
    with torch.no_grad():
      self.feat_model.eval()
      self.inlier_model.eval()
      self.val_data_loader.dataset.reset_seed(0)

      num_data = 0
      loss_meter = AverageMeter()
      hit_ratio_meter = AverageMeter()
      regist_succ_meter = AverageMeter()
      regist_rte_meter = AverageMeter()
      regist_rre_meter = AverageMeter()
      average_valid_meter = AverageMeter()
      data_timer = Timer()
      feat_timer = Timer()
      inlier_timer = Timer()
      nn_timer = Timer()
      dgr_timer = Timer()

      tot_num_data = len(self.val_data_loader.dataset)
      if self.val_max_iter > 0:
        tot_num_data = min(self.val_max_iter, tot_num_data)
      tot_num_data = int(tot_num_data / self.val_data_loader.batch_size)
      data_loader_iter = self.val_data_loader.__iter__()

      tp, fp, tn, fn = 0, 0, 0, 0
      for batch_idx in range(tot_num_data):
        data_timer.tic()
        input_dict = self.get_data(data_loader_iter)
        data_timer.toc()

        pred_rots, pred_trans, ws, weights, logits, total_matches, bil_agree_matches, time_records = self.inference(input_dict)

        gt_rots, gt_trans = self.decompose_rotation_translation(input_dict['T_gt'])
        rot_error = batch_rotation_error(pred_rots, gt_rots) * 180 / np.pi
        trans_error = batch_translation_error(pred_trans, gt_trans)

        individual_loss = rot_error + self.config.trans_weight * trans_error

        # Select batches with at least 10 valid correspondences
        valid_mask = ws >= self.ws_thresh
        num_valid = valid_mask.sum().item()

        # Registration loss against registration GT
        loss = self.config.procrustes_loss_weight * individual_loss[valid_mask].mean()

        is_correct = find_correct_correspondence(input_dict['correspondences'], total_matches, len_batch=input_dict['len_batch'])
        target = torch.from_numpy(is_correct).squeeze()
        if self.config.inlier_use_direct_loss:
            inlier_loss = self.config.inlier_direct_loss_weight * self.crit(
                logits.cpu().squeeze(), target.to(torch.float)) / self.iter_size
            loss += inlier_loss
        loss_meter.update(loss.item())

        regist_rre_meter.update(rot_error.squeeze())
        regist_rte_meter.update(trans_error.squeeze())

        # Compute success
        success = (trans_error < self.config.success_rte_thresh) * (
            rot_error < self.config.success_rre_thresh) * valid_mask
        
        regist_succ_meter.update(success.float())
        #logging.info(f'pred_rots={pred_rots}\ngt_rots={gt_rots}\nre:{rot_error.squeeze()}\nte:{trans_error.squeeze()}\nsum:{regist_succ_meter.sum}, count:{regist_succ_meter.count}')
        
        is_correct_fot_hit_ratio = find_correct_correspondence(input_dict['correspondences'], bil_agree_matches, len_batch=input_dict['len_batch'])
        hit_ratio_meter.update(is_correct_fot_hit_ratio.sum().item() / len(is_correct_fot_hit_ratio))

        feat_time, nn_time, inlier_time, dgr_time = time_records
        average_valid_meter.update(num_valid)
        feat_timer.update(feat_time)
        nn_timer.update(nn_time)
        inlier_timer.update(inlier_time)

        neg_target = (~target).to(torch.bool)
        pred = weights > 0.5  # TODO thresh
        pred_on_pos, pred_on_neg = pred[target], pred[neg_target]
        tp += pred_on_pos.sum().item()
        fp += pred_on_neg.sum().item()
        tn += (~pred_on_neg).sum().item()
        fn += (~pred_on_pos).sum().item()

        num_data += 1
        torch.cuda.empty_cache()

        if batch_idx % self.config.stat_freq == 0:
          precision = tp / (tp + fp + eps)
          recall = tp / (tp + fn + eps)
          f1 = 2 * (precision * recall) / (precision + recall + eps)
          tpr = tp / (tp + fn + eps)
          tnr = tn / (tn + fp + eps)
          balanced_accuracy = (tpr + tnr) / 2
          logging.info(' '.join([
              f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
              f"NN search time: {nn_timer.avg:.3f}, average num valid:{average_valid_meter.avg:.3f}",
              f"Feature Extraction Time: {feat_timer.avg:.3f}, Inlier Time: {inlier_timer.avg:.3f},",
              f"Loss: {loss_meter.avg:.4f}, Hit Ratio: {hit_ratio_meter.avg:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ",
              f"TPR: {tpr:.4f}, TNR: {tnr:.4f}, BAcc: {balanced_accuracy:.4f}, ",
              f"DGR RTE: {regist_rte_meter.avg:.3f}, DGR RRE: {regist_rre_meter.avg:.3f}, DGR Time: {dgr_timer.avg:.3f}",
              f"DGR Succ rate: {regist_succ_meter.avg:3f}",
          ]))
          data_timer.reset()

      precision = tp / (tp + fp + eps)
      recall = tp / (tp + fn + eps)
      f1 = 2 * (precision * recall) / (precision + recall + eps)
      tpr = tp / (tp + fn + eps)
      tnr = tn / (tn + fp + eps)
      balanced_accuracy = (tpr + tnr) / 2

      logging.info(' '.join([
          f"Feature Extraction Time: {feat_timer.avg:.3f}, NN search time: {nn_timer.avg:.3f}",
          f"Inlier Time: {inlier_timer.avg:.3f}, Final Loss: {loss_meter.avg}, ",
          f"Loss: {loss_meter.avg}, Hit Ratio: {hit_ratio_meter.avg:.4f}, Precision: {precision}, Recall: {recall}, F1: {f1}, ",
          f"TPR: {tpr}, TNR: {tnr}, BAcc: {balanced_accuracy}, ",
          f"RTE: {regist_rte_meter.avg:.3f}, RRE: {regist_rre_meter.avg:.3f}, DGR Time: {dgr_timer.avg:.3f}",
          f"DGR Succ rate: {regist_succ_meter.avg:3f}",
      ]))

      stat = {
          'loss': loss_meter.avg,
          'precision': precision,
          'recall': recall,
          'tpr': tpr,
          'tnr': tnr,
          'balanced_accuracy': balanced_accuracy,
          'f1': f1,
          'regist_rte': regist_rte_meter.avg,
          'regist_rre': regist_rre_meter.avg,
          'succ_rate': regist_succ_meter.avg
      }

    return stat

  def _load_weights(self, config):
    if config.resume is None and config.weights:
      logging.info("=> loading weights for inlier model '{}'".format(config.weights))
      checkpoint = torch.load(config.weights)
      self.feat_model.load_state_dict(checkpoint['state_dict'])
      logging.info("=> Loaded base model weights from '{}'".format(config.weights))
      if 'state_dict_inlier' in checkpoint:
        self.inlier_model.load_state_dict(checkpoint['state_dict_inlier'])
        logging.info("=> Loaded inlier weights from '{}'".format(config.weights))
      else:
        logging.warn("Inlier weight not found in '{}'".format(config.weights))

    if config.resume is not None:
      if osp.isfile(config.resume):
        logging.info("=> loading checkpoint '{}'".format(config.resume))
        state = torch.load(config.resume)

        self.start_epoch = state['epoch'] + 1
        self.feat_model.load_state_dict(state['state_dict'])
        self.feat_model = self.feat_model.to(self.device)
        self.scheduler.load_state_dict(state['scheduler'])
        self.optimizer.load_state_dict(state['optimizer'])

        if 'best_val' in state.keys():
          self.best_val = state['best_val']
          self.best_val_epoch = state['best_val_epoch']
          self.best_val_metric = state['best_val_metric']

        if 'state_dict_inlier' in state:
          self.inlier_model.load_state_dict(state['state_dict_inlier'])
          self.inlier_model = self.inlier_model.to(self.device)
        else:
          logging.warn("Inlier weights not found in '{}'".format(config.resume))
      else:
        logging.warn("Inlier weights does not exist at '{}'".format(config.resume))

  def _save_checkpoint(self, epoch, filename='checkpoint'):
    """
    Saving checkpoints

    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
    """
    print('_save_checkpoint from inlier_trainer')
    state = {
        'epoch': epoch,
        'state_dict': self.feat_model.state_dict(),
        'state_dict_inlier': self.inlier_model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'config': self.config,
        'best_val': self.best_val,
        'best_val_epoch': self.best_val_epoch,
        'best_val_metric': self.best_val_metric
    }
    filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
    logging.info("Saving checkpoint: {} ...".format(filename))
    torch.save(state, filename)

  def get_data(self, iterator):
    while True:
      try:
        input_data = iterator.next()
      except ValueError as e:
        logging.info('Skipping an empty batch')
        continue

      return input_data

  def decompose_by_length(self, tensor, reference_tensors):
    decomposed_tensors = []
    start_ind = 0
    for r in reference_tensors:
      N = len(r)
      decomposed_tensors.append(tensor[start_ind:start_ind + N])
      start_ind += N
    return decomposed_tensors

  def decompose_rotation_translation(self, Ts):
    Ts = Ts.float()
    Rs = Ts[:, :3, :3]
    ts = Ts[:, :3, 3]

    Rs.require_grad = False
    ts.require_grad = False

    return Rs, ts
  # 不带SGD refine和safe guard
  def weighted_procrustes(self, xyz0s, xyz1s, pred_pairs, weights):
    decomposed_weights = self.decompose_by_length(weights, pred_pairs)
    RT = []
    ws = []

    for xyz0, xyz1, pred_pair, w in zip(xyz0s, xyz1s, pred_pairs, decomposed_weights):
      xyz0.requires_grad = False
      xyz1.requires_grad = False
      ws.append(w.sum().item())
      predT = GlobalRegistration.weighted_procrustes(
          X=xyz0[pred_pair[:, 0]].to(self.device),
          Y=xyz1[pred_pair[:, 1]].to(self.device),
          w=w,
          eps=np.finfo(np.float32).eps)
      RT.append(predT)

    Rs, ts = list(zip(*RT))
    Rs = torch.stack(Rs, 0)
    ts = torch.stack(ts, 0)
    ws = torch.Tensor(ws)
    return Rs, ts, ws

  def find_pairs(self, xyz0s, xyz1s, F0, F1, len_batch, k, nn_max_n, return_distance, search_method):
    # 对每一个F0 find一个最近的F1
    nn_batch = find_knn_batch(F0,
                              F1,
                              len_batch,
                              nn_max_n=nn_max_n,
                              knn=k,
                              return_distance=return_distance,
                              search_method=search_method)
    pred_pairs = []
    # len(nns) == F0.shape[0]
    for xyz0, xyz1, nns, lens in zip(xyz0s, xyz1s, nn_batch, len_batch):
      # try:
      #   c = xyz1[nns]
      # except Exception as e:
      #   print(e)
      #   print(f'xyz0={xyz0.shape}, xyz1={xyz1.shape}, lens={lens}')
      #   traceback.print_exc()
      pred_pair_ind0, pred_pair_ind1 = torch.arange(
          len(nns)).long()[:, None], nns.long().cpu()
      nn_pairs = []
      # 处理knn k>1情况
      for j in range(nns.shape[1]):
        nn_pairs.append(
            torch.cat((pred_pair_ind0.cpu(), pred_pair_ind1[:, j].unsqueeze(1)), 1))
      pred_pairs.append(torch.cat(nn_pairs, 0))
    return pred_pairs

  def find_pairs_bilateral(self, F0, F1, len_batch, k, return_distance):
      # 对每一个F0 find一个最近的F1
      from extension.knn import knnModule
      knnfunc = knnModule()
      start0, start1 = 0, 0
      #[b, c, n] [b, c, m]  
      total_matches = []
      bil_agree_matches = []
      
      for lens in len_batch:
        N0, N1 = lens
        hashseed = max(N0, N1)
        data1 = F0[start0:start0 + N0].transpose(1, 0).unsqueeze(0)
        data2 = F1[start1:start1 + N1].transpose(1, 0).unsqueeze(0)
        idx1, idx2 = knnfunc(data1, data2, k, bilateral=True, return_distance=False, return_index=True)
        idx1, idx2 = idx1.squeeze(0).transpose(1, 0), idx2.squeeze(0).transpose(1, 0)
        # 0->1
        pred_pair_ind0, pred_pair_ind1 = torch.arange(
            len(idx1)).long()[:, None], idx1.long().cpu()
        nn_pairs01 = []
        # 处理knn k>1情况
        for j in range(pred_pair_ind1.shape[1]):
          nn_pairs01.append(
              torch.cat((pred_pair_ind0.cpu(), pred_pair_ind1[:, j].unsqueeze(1)), 1))
        nns01 = torch.cat(nn_pairs01, 0)
        # 0<-1
        pred_pair_ind0, pred_pair_ind1 = idx2.long().cpu(),torch.arange(len(idx2)).long()[:, None]
        # 处理knn k>1情况
        nn_pairs10 = []
        for j in range(pred_pair_ind0.shape[1]):
          nn_pairs10.append(
              torch.cat((pred_pair_ind0[:, j].unsqueeze(1), pred_pair_ind1.cpu()), 1))
        nns10 = torch.cat(nn_pairs10, 0)
        # delete dulplicate
        hash_vec01 = _hash(nns01.cpu().numpy(), hashseed)
        hash_vec10 = _hash(nns10.cpu().numpy(), hashseed)
        mask = np.isin(hash_vec10, hash_vec01)
        mask_not = np.logical_not(mask)
        totalnns = torch.cat((nns01, nns10[mask_not, :]), 0)
        total_matches.append(totalnns)
        bil_agree_matches.append(nns10[mask, :])
        start0 += N0
        start1 += N1
        
      return total_matches, bil_agree_matches
