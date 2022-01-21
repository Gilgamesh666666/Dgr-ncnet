

export THREED_MATCH_DIR="/home/zebai/datasets/FCGF_3dmatch_data/threedmatch"
export FCGF_WEIGHTS="2019-08-16_19-21-47.pth"
export PATH_POSTFIX=$1
export MISC_ARGS=$2

export DATA_ROOT="./outputs/Experiment2"
export DATASET=${DATASET:-ThreeDMatchPairDataset03}
export THREED_MATCH_DIR=${THREED_MATCH_DIR}
export MODEL=${MODEL:-ResUNetBN2C}
export MODEL_N_OUT=${MODEL_N_OUT:-32}
#export FCGF_WEIGHTS=${FCGF_WEIGHTS:fcgf.pth}
export INLIER_MODEL=${INLIER_MODEL:-ResUNetBNF}
export OPTIMIZER=${OPTIMIZER:-SGD}
export LR=${LR:-1e-1}
export MAX_EPOCH=${MAX_EPOCH:-100}
export BATCH_SIZE=${BATCH_SIZE:-32}
export ITER_SIZE=${ITER_SIZE:-1}
export VOXEL_SIZE=${VOXEL_SIZE:-0.05}
export POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER=${POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER:-4}
export CONV1_KERNEL_SIZE=${CONV1_KERNEL_SIZE:-7}
export EXP_GAMMA=${EXP_GAMMA:-0.99}
export RANDOM_SCALE=${RANDOM_SCALE:-True}
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export VERSION=$(git rev-parse HEAD)

export OUT_DIR=${DATA_ROOT}/${DATASET}-v${VOXEL_SIZE}/${INLIER_MODEL}/${OPTIMIZER}-lr${LR}-e${MAX_EPOCH}-b${BATCH_SIZE}i${ITER_SIZE}-modelnout${MODEL_N_OUT}${PATH_POSTFIX}/${TIME}

export PYTHONUNBUFFERED="True"
export WS_THRESH=${WS_THRESH:-0.0}
#echo $OUT_DIR

mkdir -m 755 -p $OUT_DIR

LOG=${OUT_DIR}/log_${TIME}.txt

# echo "Host: " $(hostname) | tee -a $LOG
# echo "Conda " $(which conda) | tee -a $LOG
# echo $(pwd) | tee -a $LOG
# echo "Version: " $VERSION | tee -a $LOG
# echo "Git diff" | tee -a $LOG
# echo "" | tee -a $LOG
# git diff | tee -a $LOG
# echo "" | tee -a $LOG
# nvidia-smi | tee -a $LOG
#--resume_dir outputs/Experiment2/ThreeDMatchPairDataset03-v0.05/ResUNetBNF/SGD-lr1e-1-e100-b8i1-modelnout32/2020-08-17_10-36-05 \
# Training
# CUDA_LAUNCH_BLOCKING=1 
CUDA_VISIBLE_DEVICES=0 python train.py \
	--inlier_knn 5 \
	--weights ${FCGF_WEIGHTS} \
	--dataset ${DATASET} \
	--threed_match_dir ${THREED_MATCH_DIR} \
	--feat_model ${MODEL} \
	--feat_model_n_out ${MODEL_N_OUT} \
	--feat_conv1_kernel_size ${CONV1_KERNEL_SIZE} \
	--ws_thresh ${WS_THRESH}\
	--inlier_model ${INLIER_MODEL} \
	--optimizer ${OPTIMIZER} \
	--lr ${LR} \
	--batch_size ${BATCH_SIZE} \
	--val_batch_size ${BATCH_SIZE} \
	--iter_size ${ITER_SIZE} \
	--max_epoch ${MAX_EPOCH} \
	--voxel_size ${VOXEL_SIZE} \
	--out_dir ${OUT_DIR} \
	--use_random_scale ${RANDOM_SCALE} \
	--clip_weight_thresh 0\
	--positive_pair_search_voxel_size_multiplier ${POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER} \
	$MISC_ARGS 2>&1 #| tee -a $LOG
	

# # Test
# python -m scripts.test_3dmatch \
# 	$MISC_ARGS \
# 	--threed_match_dir ${THREED_MATCH_DIR} \
# 	--weights ${OUT_DIR}/best_val_checkpoint.pth \
# 	2>&1 | tee -a $LOG
