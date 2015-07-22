## Genrate bounding boxes from feat pyramids
from __future__ import division
import os
import sys
import operator
import numpy
import scipy
import scipy.misc
import yaml
sys.path.append("misc")
from util import save_bbox, load_gtbox, mkdir, break_filename, get_src_filenames, plot_bbox
from nms import nms_slow
from hnm import hard_neg_mining

with open('config.yaml', 'r') as f:
    config = yaml.load(f)
dataset_dir = config['dataset_dir']
if 'data-USA' in dataset_dir:
    pyra_widths = config['data-USA_pyra_widths']
    num_pyra_lv = len(pyra_widths)
else:
    raise ValueError()
print "Dataset:", dataset_dir


# Get filenames of src imgs and feat pyramids
src_img_filenames = sorted(get_src_filenames(dataset_dir, config['phase']))
assert len(src_img_filenames)!=0

cascade_dir = config['cascade']+'/'
dir_prefix = dataset_dir+cascade_dir+config['exp_id']+'_'+config['phase']+'/'
print "Generating bounding boxes for", config['cascade']

src_feat_exp_dir = dir_prefix+'feat/' 
tar_exp_dir = dir_prefix + 'bbox/'
tar_plot_dir = dir_prefix+'plot/'    
tar_exp_dir += 'sc_' + str(config['bbox_score_thres'])
tar_exp_dir += '_nms'+str(config['nms_thres'])+'/'
mkdir(tar_exp_dir)
mkdir(tar_plot_dir)

input_h = config['input_h']
input_w = config['input_w']
pad_h = config['pad_h']
pad_w = config['pad_w']

# Generate bounding boxes for each frame
cnt = 0
for src_img_filename in src_img_filenames:
    cnt += 1
    src_img = scipy.misc.imread(src_img_filename)
    set_name, V_name, frame_name, frame_num = break_filename(src_img_filename, dataset_dir)
    bbox_lst = list()

    for lv in range(num_pyra_lv):
        feat = numpy.load(src_feat_exp_dir+str(lv)+'/'+set_name+'/'+V_name+'/'+frame_name+'.npy')
        feat_w = feat.shape[1]
        feat_h = feat.shape[0]
        src_img_w = src_img.shape[1]
        src_img_h = src_img.shape[0]
        bbox_sc_ratio =  pyra_widths[lv] / src_img_w
        bbox_h =  input_h / bbox_sc_ratio
        bbox_w = input_w / bbox_sc_ratio

        dist_w_ratio = src_img_w / (feat_w+pad_w*2) 
        dist_h_ratio = src_img_h / (feat_h+pad_h*2)
        # PASCAL format: (i, j, width, height, sc)
        for i in range(feat.shape[1]):
            for j in range(feat.shape[0]):
                left_i = (i+pad_w+0.5) * dist_w_ratio
                top_j = (j+pad_h+0.5) * dist_h_ratio
                bbox_lst.append(tuple((left_i-bbox_w/2, top_j-bbox_h/2, bbox_w, bbox_h, feat[j, i])))

    bbox_lst.sort(key=operator.itemgetter(4), reverse=True)
    tmp_lst = [sc for sc in bbox_lst if sc[4] >= config['bbox_score_thres']]
    if config['nms_on']:
        bbox_lst = nms_slow(tmp_lst, config['nms_thres'])
    else:
        bbox_lst = tmp_lst
    save_bbox(tar_exp_dir+'/'+set_name, frame_num, bbox_lst, V_name, dataset_dir)

    # Printing stuff
    if int(cnt/config['print_iter']) == (cnt/config['print_iter']):
        print set_name, V_name, frame_name, "num of bboxes after nms:", len(bbox_lst)
    if config['print_max_score_on']:
        try:
            print "MAX", bbox_lst[0]
        except IndexError:
            pass

    # Get ground truth annotations for the current frame
    if 'data-USA' in dataset_dir:
        gt_lst = load_gtbox(config['gt_dir']+'/'+set_name+'/'+V_name+'/', frame_num)
    else:
        raise ValueError()

    # Plotting
    if config['plot_bbox_on']:
        plot_bbox(src_img_filename, tar_plot_dir, bbox_lst, gt_lst)

    # Hard negative mining
    if config['hard_neg_mining_on']:
        hard_neg_mining(src_img_filename, bbox_lst, gt_lst, '/home/ryan/ccf/data/Caltech/train01/hard_mining/', config)

"""
    # Save "good" patches and dt/gt location 
    if BB_REG:
        # gt_lst = load_gtbox(GT_PATH, int(src_filename[-9:-4]))
        bb_reg_prepare_input(src_filename, dt_lst, gt_lst, tar_dir=BB_REG_PATH, overlap_thres=BB_REG_THRES)
"""
