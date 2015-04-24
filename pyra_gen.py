from __future__ import division

import os
import os.path
import sys
sys.path.insert(0, '../python')

import caffe
from caffe.proto import caffe_pb2
import numpy as np
import scipy
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import glob
import cv2
from net_init import fullconv_net_init

## Generate Heatmap Pyramids
dir = 'eth'
phase = 'test'
PLOT_ON = False
caffemodel = 'hard_mining_round2_iter_500.caffemodel'
mean_file = 'vgg_mean.npy'
res_path = dir + '/15041700_'+phase+'_res/'
if not os.path.exists(res_path):
    os.makedirs(res_path)
res_sc = 21

if dir=='eth':
    if phase=='test':
        setnames = sorted(glob.glob('/home/ryan/data/caltech/data-ETH/images/set*'))[:]
    else:
        setnames = sorted(glob.glob('/home/ryan/data/caltech/data-ETH/images/set*'))[1:]
    Vnames = []
    for setname in setnames:
        Vnames += sorted(glob.glob(setname+'/V*'))
    filenames = []
    for Vname in Vnames:
        filenames += sorted(glob.glob(Vname+'/*png'))
else:
    raise ValueError()

net_full_conv = fullconv_net_init('vgg_deploy.prototxt', 'vgg_deploy_fullconv.prototxt', caffemodel, mean_file, device_id=1, dir=dir)

src_scales = [[3402], [2700], [2143,1701], [1350,1071], [850,675,536,425]]
patch_num_lst = [5, 4, 3, 2, 1]

for bg_level in range(len(patch_num_lst)):

    bg_sc = patch_num_lst[bg_level]*650 + 200
    for src_level in range(len(src_scales[bg_level])):
        src_sc = src_scales[bg_level][src_level]
        sc_dir = res_path + str(bg_level)+'_'+str(src_level)
        if not os.path.exists(sc_dir):
            os.makedirs(sc_dir)
        for filename in filenames:
            file_dir = sc_dir + '/' + filename[-21:-4]
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            img = np.zeros((bg_sc, bg_sc, 3))
            tmp = caffe.io.load_image(filename)
            tmp_s = tmp.shape
            sc_ratio = src_sc/max(tmp_s[0]*0.41, tmp_s[1])
            tar = cv2.resize(tmp, (0,0), fx=sc_ratio, fy=sc_ratio*0.41)
            tar_s = tar.shape
            img[ 0:tar_s[0], 0:tar_s[1], :] = tar
            bg_map = np.zeros((res_sc*patch_num_lst[bg_level], res_sc*patch_num_lst[bg_level]))

            for i in range(patch_num_lst[bg_level]):
                for j in range(patch_num_lst[bg_level]):
                    offset_top = i*650
                    offset_left = j*650
                    patch = img[offset_top:offset_top+850, offset_left:offset_left+850, :] 
                    # skip the patch if it's all 0
                    # if not np.any(patch):
                    #    continue
                    out = net_full_conv.forward_all(data=np.asarray(\
                            [net_full_conv.preprocess('data', patch)]))
                    res_map = out['fc8_inria_conv'][0][1]
                    # print res_map.shape
                    savename = file_dir + '/' + 'offt_'+str(offset_top)+\
                                                        '_offl_'+str(offset_left)+'_.npy'
                    np.save(savename, res_map)
                    print "Saved to " + savename

                    bg_map[i*res_sc:(i+1)*res_sc, j*res_sc:(j+1)*res_sc] = res_map

            if PLOT_ON:
                # plot response maps
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.subplot(1, 2, 2)
                plt.imshow(bg_map)
                plt.savefig(savename+'_res.png')
                plt.close()
                # plt.show()


