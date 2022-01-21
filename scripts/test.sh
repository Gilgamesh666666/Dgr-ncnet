#! /usr/bin/bash
###
 # @Author: your name
 # @Date: 2020-11-27 20:41:25
 # @LastEditTime: 2020-12-30 21:39:10
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /exp2/scripts/test.sh
### 
# all use the voxel_size save by the checkpoint

est_trans_save_dir='old_eval_3dmatch_3rd_train_result_2'
test_save_dir='old_dgr_eval_3dmatch_train_result_2'

# CUDA_VISIBLE_DEVICES=0 python -m scripts.test_rr_3dmatch --weights ResUNetBN2C-feat32-3dmatch-v0.05.pth --test_save_dir ${est_trans_save_dir} --threed_match_dir /home/zebai/datasets/3DMatch/testdata --inlier_knn 2

#CUDA_VISIBLE_DEVICES=0 python -m scripts.test_rr_kitti --weights scripts/ResUNetBN2C-feat32-kitti-v0.3.pth --kitti_dir /home/Gilgamesh/Datasets/KITTI

# CUDA_VISIBLE_DEVICES=0 python -m scripts.test_rr_eth --weights ResUNetBN2C-feat32-3dmatch-v0.05.pth --threed_match_dir /home/zebai/datasets/ETH


# python -m scripts.test_3dmatch_eth --estRoot ${est_trans_save_dir} --gtRoot /home/zebai/test_all/3DMatch/gt_info/3dmatch --rot_thresh 15 --translate_thresh 0.3 --rmse_thresh 0.2 --scene_file scripts/test_3dmatch.txt --saveRoot ${test_save_dir} --logfile dgr-ncn_eval_3dmatch_result.log

# python test_3dmatch_eth.py --estRoot eval_result_eth --gtRoot /home/zebai/test_all/ETH/gt_info/ETH_0_018 --rot_thresh 5 --translate_thresh 2 --rmse_thresh 2 --scene_file test_eth.txt --saveRoot dgr-ncn_eval_eth_result --logfile dgr-ncn_eval_eth_result.log