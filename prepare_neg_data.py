import sys
sys.path.append('misc')
import numpy
import yaml
import glob
import scipy
import scipy.io
import scipy.misc
from util import crop_patch, get_IoU, mkdir

with open('config.yaml', 'r') as f:
    config = yaml.load(f)
acf_anno_dir = config['acf_anno_dir']
acf_data_dir = config['acf_data_dir']
acf_tar_dir = config['acf_tar_dir']
mkdir(acf_tar_dir)

img_filenames = sorted(glob.glob(acf_data_dir+'*jpg'))
matfile = scipy.io.loadmat(acf_anno_dir+'dt.mat')
dt = matfile['dt']
matfile = scipy.io.loadmat(acf_anno_dir+'gt.mat')
gt = matfile['gt']


for i in range(dt.shape[1]):
    img = scipy.misc.imread(img_filenames[i])
    dt_bbox_mat = dt[0,i] # x,y,w,h,sc,blah; numpy
    gt_bbox_mat = gt[0,i] # a list of 1x5 could be empty
    cnt = 0
    for dt_bbox in dt_bbox_mat:
        matched = False
        for gt_bbox in gt_bbox_mat:
            IoU = get_IoU(gt_bbox, dt_bbox)
            if IoU>0.2:
                matched = True
        if not matched:
            cropped_patch = crop_patch(img, dt_bbox, config['non_cropped_input_h'], config['non_cropped_input_w'])
            patch_name = acf_tar_dir+str(i)+'_'+str(cnt)+'.jpg'
            if cropped_patch.any():
                scipy.misc.imsave(patch_name, cropped_patch)
        cnt += 1
