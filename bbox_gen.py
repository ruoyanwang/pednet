## From score maps to bounding boxes
from __future__ import division
import os
import sys
sys.path.append("./pyimagesearch")
from pyimagesearch.nms import non_max_suppression_slow
import glob
import operator
import numpy
import scipy
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
import commands
from util import save_bbox, save_bbox_caltech, load_gtbox, sc2inria, inria2sc
from hard_mining import hard_mining
from bb_regression import bb_reg_prepare_input

# select TOP_K highest scores if use_top_k is True
# select scores higher than TOP_V if use_top_k is False

phase = 'train'

HARD_MINING = True
if phase=='train' and HARD_MINING:
    round_no = '0' # round number of hard_mining

BB_REG = False
BB_REG_THRES = 0.5

use_top_k = False
PLOT_ON = False


dataset = 'caltech/data-USA'
RES_PATH = './caltech/15040800_'+phase+'_res/' # don't change this
TAR_PATH = './caltech/15040800_'+phase+'/' 
TAR_CALTECH_PATH = '/home/ryan/data/caltech/data-USA/res/Ours/'
GT_PATH = '/home/ryan/data/caltech/data-USA/annotations/'
INPUT_H = 156
INPUT_W = 64

PATCH_SIZE = 800
PAD_H = 2  # depending on the network prototxt!
PAD_W = 0.5

res_sc_h = 21
res_sc_w = 24

patch_num_lst = [3, 2, 1]
bg_scales = list()
for i in range(len(patch_num_lst)):
    bg_sc_h = patch_num_lst[i] * 640 + 160
    bg_sc_w = patch_num_lst[i] * 740 + 60
    bg_scales.append((bg_sc_h, bg_sc_w))
print bg_scales
src_scales = [[1880], [1450, 1110, 855], [660, 500, 390, 300]] 


BB_REG_PATH = TAR_PATH + 'bb_reg/'

NMS_THRES = 0.25
if use_top_k:
    TOP_K = 1 # increase/enlarge boxes
    TAR_PATH += 'K' + str(TOP_K)
else:
    TOP_V = 3.3
    TAR_PATH += 'V' + str(TOP_V)
TAR_PATH += '_T'+str(NMS_THRES)
if not os.path.exists(TAR_PATH):
    os.makedirs(TAR_PATH)
if BB_REG and not os.path.exists(BB_REG_PATH):
    os.makedirs(BB_REG_PATH)


if dataset=='caltech/data-USA':
    commands.getstatusoutput('rm -r /home/ryan/data/caltech/data-USA/res/Ours/*')
    if phase=='test':
        commands.getstatusoutput('rm -r /home/ryan/data/caltech/results/UsaTest')
        setnames = sorted(glob.glob('/home/ryan/data/caltech/data-USA/images/set*'))[6:]
    else:
        commands.getstatusoutput('rm -r /home/ryan/data/caltech/results/UsaTrain')
        setnames = sorted(glob.glob('/home/ryan/data/caltech/data-USA/images/set*'))[:6]
    Vnames = []
    for setname in setnames:
        Vnames += sorted(glob.glob(setname+'/V*'))
    src_filenames = []
    for Vname in Vnames:
        src_filenames += sorted(glob.glob(Vname+'/*jpg'))
elif dataset=='inria':
    src_filenames = sorted(glob.glob('/home/ryan/data/inria/'+phase+'/pos/*png'))
else:
    raise ValueError()


for src_filename in src_filenames:
    if dataset=='caltech/data-USA':
        set_name = src_filename[-21:-16]
        V_name = src_filename[-15:-11]
        frame_name = src_filename[-10:-4]
        frame_num = int(frame_name[1:])
    else:
        frame_num = int(src_filename[-9:-4])

    # if int(set_name[-1])<=3:
    #     continue

    src_img = scipy.misc.imread(src_filename)
    init_score_lst = list()
    for bg_level in range(len(src_scales)):
        for src_level in range(len(src_scales[bg_level])):
            if dataset=='inria':
                res_map_filenames = glob.glob(RES_PATH+str(bg_level)+'_' +str(src_level)+'/'+src_filename[-9:-4]+'/*.npy')
            else:
                res_map_filenames = glob.glob(RES_PATH+str(bg_level)+'_' +str(src_level)+'/'+set_name+'/'+V_name+'/'+frame_name+'/*.npy')

            for res_map_filename in res_map_filenames:
                res_map = numpy.load(res_map_filename)
                offset_top = int(res_map_filename.split('_')[-4])
                offset_left = int(res_map_filename.split('_')[-2])
                src_img_h = src_img.shape[0]
                src_img_w = src_img.shape[1]
                sc_ratio = src_scales[bg_level][src_level] / src_img_w
                half_h = src_img_w * INPUT_H / src_scales[bg_level][src_level] * 0.5 # box; the 'w' is correct!
                half_w = half_h * 64./156
                for i in range(res_sc_h):
                    for j in range(res_sc_w):
                        # score tuple: (i, j, half_height, half_width, score)
                        # i is the vertical index, counted from up to down
                        top_i = (offset_top + (i+PAD_H+0.5) * PATCH_SIZE/(res_sc_h+PAD_H*2)) / sc_ratio
                        left_j = (offset_left + (j+PAD_W+0.5) * PATCH_SIZE/(res_sc_w+PAD_W*2))/sc_ratio
                        init_score_lst.append(tuple((int(top_i)
                                  , int(left_j), int(half_h), int(half_w), res_map[i, j])))
    init_score_lst.sort(key=operator.itemgetter(4), reverse=True)
    """
    try:
        print "MAX", init_score_lst[0]
    except IndexError:
        pass
    """
    if use_top_k:
        score_lst = init_score_lst[:TOP_K]
    else:
        score_lst = [sc for sc in init_score_lst if sc[4] >= TOP_V]

 
    ## transform bboxes to INRIA format and perform non-maximum suppresion
    score_lst_inria = sc2inria(score_lst)
    dt_lst = non_max_suppression_slow(score_lst_inria, NMS_THRES)
    filtered_score_lst = inria2sc(dt_lst)

    print "Number of bboxes:", len(filtered_score_lst)
    # save bounding boxes in INRIA&Caltech format
    if dataset=='inria':
        save_bbox(TAR_PATH+'/bbox', frame_num, filtered_score_lst )
        save_bbox_caltech(TAR_CALTECH_PATH, frame_num, filtered_score_lst)
    else:
        save_bbox_caltech(TAR_CALTECH_PATH+'/'+set_name, frame_num, filtered_score_lst, V_name, dataset)

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
