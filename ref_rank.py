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
import leveldb
import caffe
from caffe.proto import caffe_pb2
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

# Load LevelDB
db = leveldb.LevelDB('ref/proposals_feat')

# Collect filenames of src imgs and bboxes
config['phase'] = 'test'
src_img_filenames = sorted(get_src_filenames(dataset_dir, config['phase']))
assert len(src_img_filenames)!=0
dir_prefix = config['src_test_bbox_dir']
tar_ref_dir = dir_prefix+config['ref']+'/' 
mkdir(tar_ref_dir)
input_h = config['input_h']
input_w = config['input_w']

# Generate bounding boxes for each frame
cnt = 0
error_rate = 0
for src_img_filename in src_img_filenames:
    # src_img = scipy.misc.imread(src_img_filename)
    set_name, V_name, frame_name, frame_num = break_filename(src_img_filename, dataset_dir)
    if '6' not in set_name:
        continue
    print src_img_filename

    bbox_lst = list()
    gt_lst = load_gtbox(dir_prefix+set_name+'/'+V_name, frame_num, VorI='V')

    for gt in gt_lst:
        datum = caffe_pb2.Datum.FromString(db.Get(str(cnt)))
        pred = caffe.io.datum_to_array(datum)
        bbox_lst.append(gt[:4]+tuple([pred[0, 1, 0]]))
        cnt += 1

    bbox_lst.sort(key=operator.itemgetter(4), reverse=True)
    print "num of bbox:", len(bbox_lst)
    tmp_lst = [sc for sc in bbox_lst if sc[4] > 0.000001]
    print "num of bbox > 0.5:", len(tmp_lst)


    # bbox_lst = nms_slow(bbox_lst, config['nms_thres']) # tmp_lst!!!!
    save_bbox(tar_ref_dir+'/'+set_name, frame_num, tmp_lst, V_name, config['dataset_dir']) # tmp!!!



"""



    # save_bbox(tar_exp_dir+'/'+set_name, frame_num, bbox_lst, V_name, dataset_dir)

    # Printing stuff
    if int(cnt/config['print_iter']) == (cnt/config['print_iter']):
        print set_name, V_name, frame_name, "num of bboxes after nms:", len(bbox_lst)
    if config['print_max_score_on']:
        try:
            print "MAX", bbox_lst[0]
        except IndexError:
            pass

    # Plotting
    if config['plot_bbox_on']:
        plot_bbox(src_img_filename, tar_plot_dir, bbox_lst, gt_lst)
        
"""
