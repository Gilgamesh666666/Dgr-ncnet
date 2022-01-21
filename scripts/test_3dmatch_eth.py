import os
import numpy as np
import open3d as o3d
import time

def est_trans_one_pair(root, scene, idx1, idx2):
    npz = np.load(os.path.join(root, scene, f'cloud_bin_{idx1}_{idx2}.npz'))
    trans = npz['trans']
    reg_time = npz['time']
    trans = np.linalg.inv(trans)
    return trans, reg_time


def registration_recall_one_pair(est_trans, gt_trans, gt_info, rmse_thresh=0.2):
    DCM = np.dot(np.linalg.inv(gt_trans), est_trans)
    qout = np.zeros(4)
    qout[0] = 0.5 * np.sqrt(1 + DCM[0, 0] + DCM[1, 1] + DCM[2, 2])
    qout[1] = - (DCM[2, 1] - DCM[1, 2]) / ( 4 * qout[0] )
    qout[2] = - (DCM[0, 2] - DCM[2, 0]) / ( 4 * qout[0] )
    qout[3] = - (DCM[1, 0] - DCM[0, 1]) / ( 4 * qout[0] )
    kc = np.concatenate((DCM[:3, 3], -qout[1:]), axis=0)[:, np.newaxis] #[6, 1]
    rmse = np.dot(np.dot(kc.T, gt_info), kc)/gt_info[0, 0]
    if rmse < rmse_thresh*rmse_thresh:
        return 1
    else:
        return 0
def RE_TE_one_pair(gt, est):
    import math
    # np [4, 4], [4, 4]
    gt_R = gt[:3, :3] # [3, 3]
    est_R = est[:3, :3] # [3, 3]
    A = (np.trace(np.dot(gt_R.T, est_R)) - 1)/2
    if A > 1:
        A = 1
    elif A < -1:
        A = -1
    rotError = math.degrees(math.fabs(math.acos(A))) # degree
    translateError = np.linalg.norm(gt[:3, 3] - est[:3, 3]) # norm
    return rotError, translateError

class Threedmatch_Log_Info:
    def __init__(self, idx1, idx2, trans, info):
        self.idx1 = idx1
        self.idx2 = idx2
        self.trans = trans # np array [4,4]
        self.info = info # np array [6,6]
    def __str__(self):
        informationstr = f'idx1:{self.idx1}, idx2:{self.idx2}\ntrans:{np.array_str(self.trans)}\ninfo:{np.array_str(self.info)}'
        return informationstr
    def __repr__(self):
        informationstr = f'idx1:{self.idx1}, idx2:{self.idx2}\ntrans:{np.array_str(self.trans)}\ninfo:{np.array_str(self.info)}'
        return informationstr
# read gt log
def read_gt_log_info(root, scene_list, suffix=''):
    log_info = {}
    for scene in scene_list:
        log_info_one_scene = {}
        scenePath = os.path.join(root, scene + suffix)
        logfile = os.path.join(scenePath, 'gt.log')
        infofile = os.path.join(scenePath, 'gt.info')
        with open(logfile, 'r') as f:
            gtlog = f.readlines()
        with open(infofile, 'r') as f:
            gtinfo = f.readlines()
        i, j = 0, 0
        while i < len(gtlog):
            logline = gtlog[i].strip().split('\t')
            assert len(logline)==3
            head = [int(s) for s in logline]
            trans = np.array([[float(s) for s in gtlog[i+1].strip().split('\t')],
                            [float(s) for s in gtlog[i+2].strip().split('\t')],
                            [float(s) for s in gtlog[i+3].strip().split('\t')],
                            [float(s) for s in gtlog[i+4].strip().split('\t')]])
            log_info_one_scene[f'{head[0]}_{head[1]}'] = Threedmatch_Log_Info(head[0], head[1], trans, None)
            i += 5
        while j < len(gtinfo):
            infoline = gtinfo[j].strip().split('\t')
            assert len(infoline)==3
            head = [int(s) for s in infoline]
            info = np.array([[float(s) for s in gtinfo[j+1].strip().split('\t')],
                            [float(s) for s in gtinfo[j+2].strip().split('\t')],
                            [float(s) for s in gtinfo[j+3].strip().split('\t')],
                            [float(s) for s in gtinfo[j+4].strip().split('\t')],
                            [float(s) for s in gtinfo[j+5].strip().split('\t')],
                            [float(s) for s in gtinfo[j+6].strip().split('\t')]])
            log_info_one_scene[f'{head[0]}_{head[1]}'].info = info
            j += 7
        log_info[scene] = log_info_one_scene
    return log_info

def evaluate_one_scene(args, log_info, scene):
    saveDir = os.path.join(args.saveRoot, scene)
    os.makedirs(saveDir, exist_ok=True)
    log_infoes = log_info[scene]
    gt_num = len(log_infoes)
    time_sum = 0
    rotError_sum = 0
    translateError_sum = 0
    registration_true_positive_sum = 0
    rot_translate_positive_sum = 0
    registration_gt_num = 0
    #registration_pred_positive_sum = 0
    for key, log_info in log_infoes.items():
        idx1 = log_info.idx1
        idx2 = log_info.idx2
        savePath = os.path.join(saveDir, f'cloud_{idx1}_{idx2}.result.txt')
        if os.path.exists(savePath):
            with open(savePath, 'r') as f:
                content = f.readlines()
            data = content[0].strip().split('\t')
            reg_time, rotError, translateError = [float(i) for i in data[:3]]
            registration_gt, registration_true_positive, rot_translate_positive = [int(i) for i in data[3:]]#, registration_pred_positive = [int(i) for i in data[5:]]
            
            time_sum += reg_time
            rotError_sum += rotError
            translateError_sum += translateError
            registration_gt_num += registration_gt
            registration_true_positive_sum += registration_true_positive
            rot_translate_positive_sum += rot_translate_positive
            #registration_pred_positive_sum += registration_pred_positive
            continue
        
        # target -> source
        est_trans, reg_time = est_trans_one_pair(args.estRoot, scene, idx1, idx2)

        gt_trans = log_info.trans
        gt_info = log_info.info
        
        # target -> source
        
        rotError, translateError = RE_TE_one_pair(gt_trans, est_trans)
        if (rotError < args.rot_thresh) and (translateError < args.translate_thresh):
            rot_translate_positive = 1
        else:
            rot_translate_positive = 0
        if (idx2 - idx1) >1:
            registration_gt = 1
            registration_true_positive = registration_recall_one_pair(est_trans, gt_trans, gt_info, args.rmse_thresh)
        else:
            registration_gt = 0
            registration_true_positive = 0
            
        # 写文件:inlier_ratio, reg_time, rotError, translateError
        # registration_true_positive, feature_matching_num, registration_pred_positive
        # est_trans
        
        with open(savePath, 'w+') as f:
            f.write(f'{reg_time:.4f}\t{rotError:.4f}\t{translateError:.4f}\t{registration_gt}\t{registration_true_positive}\t{rot_translate_positive}\n')#{registration_pred_positive}\n')
            
        
        time_sum += reg_time
        rotError_sum += rotError
        translateError_sum += translateError
        registration_gt_num += registration_gt
        registration_true_positive_sum += registration_true_positive
        rot_translate_positive_sum += rot_translate_positive
        #registration_pred_positive_sum += registration_pred_positive

    time_avg = time_sum/gt_num
    rotError_avg = rotError_sum/gt_num
    translateError_avg = translateError_sum/gt_num
    registration_recall = registration_true_positive_sum/registration_gt_num
    rot_translate_recall = rot_translate_positive_sum/gt_num

    return time_avg, rotError_avg, translateError_avg, \
            registration_recall, rot_translate_recall#, registration_precision

# TODO: read feature file
# rootdir write in the function
# to be implement to read the features 
# for every point cloud in 3dmatch test set

def main():
    import argparse
    import logging
    from multiprocessing import Pool, cpu_count
    from functools import partial
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gtRoot', default=None, type=str)
    parser.add_argument('--saveRoot', default=None, type=str)
    parser.add_argument('--logfile', default=None, type=str)
    parser.add_argument('--rot_thresh', default=15, type=float)
    parser.add_argument('--translate_thresh', default=0.3, type=float)
    parser.add_argument('--rmse_thresh', default=0.2, type=float)
    parser.add_argument('--estRoot', default=None, type=str)
    parser.add_argument('--scene_file', default=None, type=str)
    args = parser.parse_args()
    if args.saveRoot is None:
        args.saveRoot = os.path.join(os.getcwd(), 'evaluate-result')
    
    BASIC_CONFIG = '[%(levelname)s] %(asctime)s:%(message)s'
    DATE_CONFIG = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(filename=args.logfile, format=BASIC_CONFIG, datefmt=DATE_CONFIG, level=logging.DEBUG)
    logging.info(f'args = {args}')

    scene_list = [scene.strip() for scene in open(args.scene_file).read().split()]
    
    log_info = read_gt_log_info(args.gtRoot, scene_list, suffix='-evaluation')
    
    pool = Pool(min(len(scene_list), cpu_count()))
    func = partial(evaluate_one_scene, args, log_info)
    results = pool.map(func, scene_list)
    pool.close()
    pool.join()

    time_avg_list, rotError_avg_list, translateError_avg_list = [], [], []
    registration_recall_list, rot_translate_recall_list = [], []
    #registration_precision_list = []
    for result in results:
        time_avg,rotError_avg, translateError_avg, \
        registration_recall, rot_translate_recall = result
        #registration_recall, registration_precision = result
        #logging.info(f'------ {scene} ------')
        
        logging.info(f"Registration Recall {registration_recall*100:.3f}%")
        logging.info(f'Rot Translate Recall {rot_translate_recall*100:.3f}%')
        
        logging.info(f"Average Reg Time: {time_avg:.3f}s")
        logging.info(f"Average degree Error: {rotError_avg:.3f} degree")
        logging.info(f"Average translate Error: {translateError_avg:.3f}")
        
        registration_recall_list.append(registration_recall)
        rot_translate_recall_list.append(rot_translate_recall)
        time_avg_list.append(time_avg)
        rotError_avg_list.append(rotError_avg)
        translateError_avg_list.append(translateError_avg)
    logging.info('-------------------- Summary ---------------------------------------')
   
    logging.info(f"All {len(scene_list)} Scenes Registration Recall {sum(registration_recall_list)*100/len(registration_recall_list):.3f}%")
    logging.info(f"All {len(scene_list)} Scenes Rot Translate Recall {sum(rot_translate_recall_list)*100/len(rot_translate_recall_list):.3f}%")
    
    logging.info(f"All {len(scene_list)} Scenes Average Reg Time: {sum(time_avg_list)/len(time_avg_list):.3f}s")
    logging.info(f"All {len(scene_list)} Scenes Average degree Error: {sum(rotError_avg_list)/len(rotError_avg_list):.3f} degree")
    logging.info(f"All {len(scene_list)} Scenes Average translate Error: {sum(translateError_avg_list)/len(translateError_avg_list):.3f}")

    logging.info(f"All {len(scene_list)} Scenes STD Registration Recall {np.std([i*100 for i in registration_recall_list]):.3f}%")
    logging.info(f"All {len(scene_list)} Scenes STD Rot Translate Recall {np.std([i*100 for i in rot_translate_recall_list]):.3f}%")
    
    logging.info(f"All {len(scene_list)} Scenes STD Average Reg Time: {np.std(time_avg_list):.3f}s")
    logging.info(f"All {len(scene_list)} Scenes STD Average degree Error: {np.std(rotError_avg_list):.3f} degree")
    logging.info(f"All {len(scene_list)} Scenes STD Average translate Error: {np.std(translateError_avg_list):.3f}")

if __name__ == '__main__':
    main()
    '''
    要把all_feat加上
    把estimate source->target然后再inv改成直接target->source
    把log trans改成广泛用的那种，不要写死
    voxel_size改个名字
    '''