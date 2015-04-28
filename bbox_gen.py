## Genrate bounding boxes from feat pyramids
from __future__ import division
import os
import sys
import glob
import operator
import numpy
import scipy
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
import commands
import yaml
sys.path.append("misc")
from util import save_bbox, load_gtbox, mkdir, break_filename, get_src_filenames
# from hard_mining import hard_mining
# from bb_regression import bb_reg_prepare_input


with open('config.yaml', 'r') as f:
    config = yaml.load(f)
dataset_dir = config['dataset_dir']
cascade_dir = config[config['cascade']+'_dir']
if 'data-USA' in dataset_dir:
    pyra_widths = config['data-USA_pyra_widths']
    num_pyra_lv = len(pyra_widths)
else:
    raise ValueError()
print "Dataset:", dataset_dir
print "Generating bounding boxes for", config['cascade']

# Get filenames of src imgs and feat pyramids
src_feat_exp_dir = dataset_dir+cascade_dir+config['exp_id']+'_'+config['phase']+'_res/' 
src_img_filenames = sorted(get_src_filenames(dataset_dir, config['phase']))
assert len(src_img_filenames)!=0

tar_exp_dir = cascade_dir+config['exp_id']+'_'+config['phase']+'_bbox/'
tar_exp_dir += 'sc_' + str(config['bbox_score_thres'])
tar_exp_dir += '_nms'+str(config['nms_thres'])+'/'
mkdir(tar_exp_dir)

input_h = config['input_h']
input_w = config['input_w']
pad_h = config['pad_h']
pad_w = config['pad_w']

for src_img_filename in src_img_filenames:
    src_img = scipy.misc.imread(src_img_filename)
    set_name, V_name, frame_name, frame_num = break_filename(src_img_filename, dataset_dir)
    bbox_lst = list()

    for lv in range(num_pyra_lv):
        feat = numpy.load(src_feat_exp_dir+str(lv)+'/'+set_name+'/'+V_name+'/'+frame_name+'.npy')
        feat_w = feat.shape[1]
        feat_h = feat.shape[0]
        src_img_w = src_img.shape[1]
        bbox_sc_ratio =  pyra_widths[lv] / src_img_w
        bbox_h =  input_h / bbox_sc_ratio
        bbox_w = input_w / bbox_sc_ratio

        dist_sc_ratio = src_img_w /(feat_w+pad_w*2)
        # PASCAL format: (i, j, width, height, sc)
        for i in range(feat.shape[1]):
            for j in range(feat.shape[0]):
                left_i = (i+pad_w+0.5) * dist_sc_ratio
                top_j = (j+pad_h+0.5) * dist_sc_ratio
                bbox_lst.append(tuple((left_i, top_j, bbox_w, bbox_h, feat[j, i])))

    bbox_lst.sort(key=operator.itemgetter(4), reverse=True)
    if config['print_max_score_on']:
        try:
            print "MAX", bbox_lst[0]
        except IndexError:
            pass

    tmp_lst = [sc for sc in bbox_lst if sc[4] >= config['bbox_score_thres']]
    # bbox_lst = non_max_suppression_slow(tmp_inria, NMS_THRES)
    bbox_lst = tmp_lst
    print "Number of bboxes:", len(bbox_lst)

    save_bbox(tar_exp_dir+'/'+set_name, frame_num, bbox_lst, V_name, dataset_dir)


"""
# load ground truth bboxes and print det results
if dataset=='inria':
    gt_lst = load_gtbox(GT_PATH, frame_num)
else:
    gt_lst = load_gtbox(GT_PATH+'/'+set_name+'/'+V_name+'/', frame_num)
if PLOT_ON:
    plt.imshow(src_img)
    ax = plt.gca()
    for sc in filtered_score_lst:
        ax.add_patch(matplotlib.patches.Rectangle((sc[1]-sc[3],sc[0]-sc[2]), sc[3]*2, sc[2]*2, alpha=1, facecolor='none', edgecolor='yellow', linewidth=2.0))
    if dataset=='inria':
        for sc in gt_lst:
            ax.add_patch(matplotlib.patches.Rectangle((sc[0],sc[1]), sc[2], sc[3]
                                  , alpha=1, facecolor='none', edgecolor='red', linewidth=2.0))

    plt.savefig(TAR_PATH + '/' + set_name + '_' + V_name + '_' + src_filename[-9:-4] + '.png')
    # plt.show()
    plt.close()

    if phase == 'train' and HARD_MINING:
        hard_mining(src_filename, dt_lst, gt_lst, tar_dir='/home/ryan/data/'+dataset+'/train/hard_mining/round'+round_no, dataset=dataset)

    # Save "good" patches and dt/gt location 
    if BB_REG:
        # gt_lst = load_gtbox(GT_PATH, int(src_filename[-9:-4]))
        bb_reg_prepare_input(src_filename, dt_lst, gt_lst, tar_dir=BB_REG_PATH, overlap_thres=BB_REG_THRES)
"""
