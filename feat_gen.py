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
dataset_dir = config['dataset_dir']
cascade0_dir = config['dataset_dir']+config['cascade0_dir']
tar_exp_dir = cascade0_dir+config['exp_id']+'_'+config['phase']+'_res/'
models_dir = config['models_dir']
# if not os.path.exists(tar_dir):
#    os.makedirs(tar_dir)

src_filenames = get_src_filenames(dataset_dir, config['phase'])

if 'data-USA' in dataset_dir:
    pyra_widths = config['data-USA_pyra_widths']
else:
    raise ValueError()
num_lv = len(pyra_widths)
print "Num of pyramid levels:", num_lv

# Generate feature pyramids
for lv in range(num_lv):
    convnet = fullconv_net_init(models_dir+config['conv_prototxt'], models_dir+config['full_conv_prototxt_prefix']+str(lv)+'.prototxt', cascade0_dir+config['cascade0_model'], config['mean_file'], config['device_id'], dataset_dir, lv)
    tar_lv_dir = tar_exp_dir + str(lv)+'/'
    for src_filename in src_filenames:
        if 'data-USA' in dataset_dir:
            tar_file_dir = tar_lv_dir+src_filename[-21:-10]
            filename_prefix = src_filename[-10:-4]
        else:
            raise ValueError()
        if not os.path.exists(tar_file_dir):
                os.makedirs(tar_file_dir)

        img = caffe.io.load_image(src_filename)
        conv_in =  [convnet.preprocess('data', img)]
        conv_out = convnet.forward_all(data=np.asarray(conv_in))
        feat = conv_out['fc8_inria_conv'][0][1]
        # print feat.shape
        savename = tar_file_dir + filename_prefix + '.npy'
        np.save(savename, feat)
        print "Saved to " + savename

        if config['plot_feat']:
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.imshow(feat)
            plt.savefig(savename+'_res.png')
            plt.close()



