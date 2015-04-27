## Generate feature pyramids
from __future__ import division
import os
import os.path
import sys
sys.path.append('../python')
sys.path.append('misc')
import caffe
from caffe.proto import caffe_pb2
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import glob
import cv2
import yaml

from net_init import fullconv_net_init
from util import get_src_filenames


# Unpack configs
with open('config.yaml', 'r') as f:
    config = yaml.load(f)
cascade0_dir = config['dataset_dir']+config['cascade0_dir']+'/'
tar_dir = cascade0_dir+config['exp_id']+'_'+config['phase']+'/'
models_dir = config['models_dir']
if not os.path.exists(tar_dir):
    os.makedirs(tar_dir)

src_filenames = get_src_filenames(config['dataset_dir'], config['phase'])

if 'data-USA' in config['dataset_dir']:
    pyra_widths = config['data-USA_pyra_widths']
else:
    raise ValueError()
num_lv = len(pyra_widths)
print "Num of pyramid levels:", num_lv

# Initialize ConvNets for all levels
convnets = list()
for lv in range(num_lv):
    convnets.append(fullconv_net_init(models_dir+config['conv_prototxt'], models_dir+config['full_conv_prototxt_prefix']+str(lv)+'.prototxt', cascade0_dir+config['cascade0_model'], config['mean_file'], config['device_id'], config['dataset_dir']))


for bg_level in range(len(patch_num_lst)):
    bg_sc_w = patch_num_lst[bg_level]*740 + 60
    bg_sc_h =  patch_num_lst[bg_level]*640 + 160
    for src_level in range(len(src_scales[bg_level])):
        src_sc = src_scales[bg_level][src_level]
        sc_dir = res_path + str(bg_level)+'_'+str(src_level)
        if not os.path.exists(sc_dir):
            os.makedirs(sc_dir)
        for filename in filenames:
            if dir=='caltech':
                file_dir = sc_dir + '/' + filename[-21:-4]
            else:
                file_dir = sc_dir + '/' + filename[-9:-4]
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            img = np.zeros((bg_sc_h, bg_sc_w, 3))
            tmp = caffe.io.load_image(filename)
            tmp_s = tmp.shape
            sc_ratio = src_sc/max(tmp_s[0], tmp_s[1])
            tar = cv2.resize(tmp, (0,0), fx=sc_ratio, fy=sc_ratio)
            tar_s = tar.shape
            img[ 0:tar_s[0], 0:tar_s[1], :] = tar
            bg_map = np.zeros((res_sc_h*patch_num_lst[bg_level], res_sc_w*patch_num_lst[bg_level]))

            for i in range(patch_num_lst[bg_level]):
                for j in range(patch_num_lst[bg_level]):
                    offset_top = i*640
                    offset_left = j*740
                    patch = img[offset_top:offset_top+800, offset_left:offset_left+800, :] 
                    # skip the patch if it's all 0
                    # if not np.any(patch):
                    #    continue
                    out = net_full_conv.forward_all(data=np.asarray(\
                            [net_full_conv.preprocess('data', patch)]))
                    res_map = out['fc8_inria_conv'][0][1]
                    print res_map.shape
                    savename = file_dir + '/' + 'offt_'+str(offset_top)+\
                                                        '_offl_'+str(offset_left)+'_.npy'
                    np.save(savename, res_map)
                    print "Saved to " + savename

                    bg_map[i*res_sc_h:(i+1)*res_sc_h, j*res_sc_w:(j+1)*res_sc_w] = res_map

            if PLOT_ON:
                # plot response maps
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.subplot(1, 2, 2)
                plt.imshow(bg_map)
                plt.savefig(savename+'_res.png')
                plt.close()
                # plt.show()


