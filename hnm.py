from __future__ import division
import operator
import os
import scipy
import scipy.misc
import numpy
import cv2

def hard_neg_mining(src_filename, dt_boxes, gt_boxes, tar_dir, config):
    """
    Save false pos/neg img patches to tar_dir

    take:
    dt_boxes: detection results (left, top, right, bottom)
    gt_boxes: ground truth boxes (left, top, width, height)

    """

    train_h = config['non_cropped_input_h']
    train_w = config['non_cropped_input_w']

    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)
    src_img = scipy.misc.imread(src_filename)

    # Sort detection boxes according to scores
    patch_cnt = 0
    for i in range(len(dt_boxes)):
        matched = False
        l0 = dt_boxes[i][0]
        t0 = dt_boxes[i][1]
        r0 = l0 + dt_boxes[i][2]
        b0 = t0 + dt_boxes[i][3]
        area0 = (r0 - l0 + 1) * (b0 - t0 + 1)

        for j in range(len(gt_boxes)):
            l1 = gt_boxes[j][0]
            t1 = gt_boxes[j][1]
            r1 = l1 + gt_boxes[j][2]
            b1 = t1 + gt_boxes[j][3]

            area1 = (r1 - l1 + 1) * (b1 - t1 + 1)

            xx1 = max(l0, l1)
            yy1 = max(t0, t1)
            xx2 = min(r0, r1)
            yy2 = min(b0, b1)

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / (area0+area1-w*h)
            if overlap>config['hard_mining_thres']:
                matched = True

        if not matched:
            try:
                patch = src_img[t0:b0, l0:r0, :]
                if 'data-USA' in config['dataset_dir']:
                    set_name = src_filename[-21:-16]
                    V_name = src_filename[-15:-11]
                    frame_name = src_filename[-10:-4]
                    savename = tar_dir+'/'+set_name+'_'+V_name+'_'+frame_name +'_'+str(patch_cnt)+'.png'
                else:
                    raise ValueError()
                if 0 in patch.shape:
                    continue
            except ValueError:
                continue
            patch_cnt += 1
            warpped_patch = cv2.resize(patch, (0,0), fx=train_w/patch.shape[1], fy=train_h/patch.shape[0])
            scipy.misc.imsave(savename, warpped_patch)
