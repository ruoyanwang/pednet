## Genrate bounding boxes from feat pyramids
from __future__ import division
import os
import sys
import glob
import numpy
import scipy
import scipy.misc
import matplotlib.pyplot as plt
import yaml
sys.path.append("misc")
from util import mkdir, break_filename, get_src_filenames


with open('config.yaml', 'r') as f:
    config = yaml.load(f)
dataset_dir = config['dataset_dir']
cascade_dir_lst = glob.glob(dataset_dir+'cascade*')
num_cascade = len(cascade_dir_lst)
print "Dataset:", dataset_dir
print "Cascade to merge", cascade_dir_lst

if 'data-USA' in dataset_dir:
    num_pyra_lv = len(config['data-USA_pyra_widths'])
else:
    raise ValueError()

# Get filenames of src imgs and feat pyramids
src_feat_exp_dir_lst = list()
for cascade_dir in cascade_dir_lst:
    src_feat_exp_dir_lst.append(cascade_dir+'/'+config['exp_id']+'_'+config['phase']+'/feat/')
src_img_filenames = sorted(get_src_filenames(dataset_dir, config['phase']))
assert len(src_img_filenames)!=0

for src_img_filename in src_img_filenames:
    set_name, V_name, frame_name, frame_num = break_filename(src_img_filename, dataset_dir)

    for lv in range(num_pyra_lv):
        for cas_no in range(num_cascade):
            feat = numpy.load(src_feat_exp_dir_lst[cas_no]+str(lv)+'/'+set_name+'/'+V_name+'/'+frame_name+'.npy')
            if cas_no==0:
                feat_w = feat.shape[1]
                feat_h = feat.shape[0]
                summed_feat = feat
            else:
                summed_feat += feat
        feat_sum_tar_dir = dataset_dir+config['feat_sum_exp_dir']+'feat/'+str(lv)+'/'+set_name+'/'+V_name+'/'
        mkdir(feat_sum_tar_dir)
        savename = feat_sum_tar_dir+frame_name + '.npy'
        numpy.save(savename, summed_feat)
        print "Saved to " + savename

        if config['plot_feat']:
            img = scipy.misc.imread(src_img_filename)
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.imshow(feat)
            plt.savefig(savename+'_res.png')
            plt.close()
                
