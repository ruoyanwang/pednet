## Genrate bounding boxes from feat pyramids
from __future__ import division
import sys
sys.path.append('../python')
sys.path.append('misc')
import glob
import operator
import numpy
import scipy
import scipy.misc
import yaml
import caffe
import random
from net_init import net_init
from util import save_bbox, load_gtbox, crop_patch, mkdir, break_filename, get_src_filenames, plot_bbox
from nms import nms_slow

with open('config.yaml', 'r') as f:
    config = yaml.load(f)
dataset_dir = config['dataset_dir']
if 'data-USA' in dataset_dir:
    pyra_widths = config['data-USA_pyra_widths']
    num_pyra_lv = len(pyra_widths)
else:
    raise ValueError()
print "Dataset:", dataset_dir

# Set up ConvNet
# convnet = net_init(config)

# Collect filenames of src imgs and bboxes
config['phase'] = 'test'
src_img_filenames = sorted(get_src_filenames(dataset_dir, config['phase']))
assert len(src_img_filenames)!=0
dir_prefix = config['src_test_bbox_dir']
tar_ref_dir = dir_prefix+'ref/' 
mkdir(tar_ref_dir)
input_h = config['input_h']
input_w = config['input_w']

# Generate bounding boxes for each frame
cnt = 0
error_rate = 0
for src_img_filename in src_img_filenames:
    print src_img_filename
    cnt = 0
    src_img = scipy.misc.imread(src_img_filename)
    set_name, V_name, frame_name, frame_num = break_filename(src_img_filename, dataset_dir)
    ###
    if '6' not in set_name:
        continue

    ###
    proposal_lst = list()
    bbox_lst = list()
    gt_lst = load_gtbox(dir_prefix+set_name+'/'+V_name, frame_num, VorI='V')
    for gt in gt_lst:
        proposal = crop_patch(src_img, gt, config['non_cropped_input_h'], config['non_cropped_input_w'])
        # proposal_lst.append(proposal)
        scipy.misc.imsave('ref/proposals/'+set_name+V_name+'_'+str(frame_num).zfill(4)+'_'+str(cnt).zfill(4)+'.png', proposal)
        cnt += 1
