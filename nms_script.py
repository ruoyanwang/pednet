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

for src_img_filename in src_img_filenames:
    print src_img_filename
    cnt += 1
    src_img = scipy.misc.imread(src_img_filename)
    set_name, V_name, frame_name, frame_num = break_filename(src_img_filename, dataset_dir)

    bbox_lst = list()
    dt_lst = load_gtbox(tar_ref_dir+set_name+'/'+V_name, frame_num, VorI='V')
    dt_lst.sort(key=operator.itemgetter(4), reverse=True)
    tmp_lst = [sc for sc in dt_lst if sc[4] >= 0.01]

    dt_lst = tmp_lst
    # dt_lst = nms_slow(dt_lst, 0.9)
    print len(dt_lst)
    save_bbox(tar_ref_dir+'/'+set_name+'_filtered', frame_num, dt_lst, V_name, config['dataset_dir'])


