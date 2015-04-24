from __future__ import division
import os
import sys
import operator
import numpy
 

def save_bbox(dir, img_idx, sc_lst):
    """
    save bboxes of an image
    PASCAL format: (w_s, h_s, w, h) // Starting from upper-left corner

    take:
    dir: saving directory
    img_idx: index number of the image
    sc_lst: score tuples (i, j, half_h, half_w, score )
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    idx = str(img_idx)
    text_name = 'I' + idx.zfill(5) + '.txt'
    print text_name
    try:
        os.remove(dir+'/'+text_name)
    except OSError:
        pass
    text = open(dir+'/'+text_name, 'w+')
    for sc in sc_lst:
        sc2w0 = str(int(sc[1]-sc[3]))
        sc2w1 = str(int(sc[0]-sc[2]))
        sc2w2 = str(int(sc[3]*2))
        sc2w3 = str(int(sc[2]*2))
        sc2w4 = str(sc[4])
        text.write(sc2w0+' '+sc2w1+' '+sc2w2+' '+sc2w3+' '+sc2w4+'\n')
    text.close()


def save_bbox_caltech(dir, img_idx, sc_lst, V_name='V000', dataset='inria'):
    """
    save bboxes of an image in a Caltech bbox file
    PASCAL format: (w_s, h_s, w, h) // Starting from upper-left corner

    take:
    dir: saving directory
    img_idx: index number of the image
    sc_lst: score tuples (i, j, half_h, half_w, score )
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    text_name = V_name + '.txt'

    if (img_idx == 0 and dataset=='inria') or (img_idx == 29 and dataset=='caltech'):
        try:
            os.remove(dir+'/'+text_name)
        except OSError:
            pass
        text = open(dir+'/'+text_name, 'w+')
    else:
        text = open(dir+'/'+text_name, 'a')

    # For Caltech format, we need to sort boxes according to its scores
    sc_lst.sort(key=operator.itemgetter(4), reverse=True)

    for sc in sc_lst:
        sc2w0 = str(int(sc[1]-sc[3]))
        sc2w1 = str(int(sc[0]-sc[2]))
        sc2w2 = str(int(sc[3]*2))
        sc2w3 = str(int(sc[2]*2))
        sc2w4 = str(sc[4])
        text.write(str(img_idx+1)+' '+sc2w0+' '+sc2w1+' '+sc2w2+' '+sc2w3+' '+sc2w4+'\n')
    text.close()

 
def load_gtbox(dir, img_idx):
    """
    load bboxes text files in PASCAL formats

    take:
    img_idx: int 
    give:
    gt_lst: a list with tuples in (left, top, width, height)
    """
    print dir, img_idx
    gt_lst = list()
    text_name = str(img_idx)
    with open(dir+'I'+text_name.zfill(5)+'.txt','r') as f:
        for line in f:
            if line.startswith('%'):
                continue
            if line.startswith('pe'):
                fr = line.split()[1:5]
                fr_t = (int(fr[0]), int(fr[1]), int(fr[2]), int(fr[3]))
            else:
                fr = line.split()[:5]
                fr_t = (float(fr[0]), float(fr[1]), float(fr[2]), float(fr[3]), float(fr[4]))
            gt_lst.append(fr_t)
    return gt_lst

def sc2inria(sc_lst):
    """
    transform a list of score tuples to INRIA format: left, right, top, bottom
    take: a list of tuples [(i ,j , hh, hw, score)]
    give: numpy arrays [(left, top, right, bottom, score), ...]
    """
    sc_lst_n = list()
    for sc in sc_lst:
        x0 = sc[1] - sc[3]
        y0 = sc[0] - sc[2]
        x1 = sc[1] + sc[3]
        y1 = sc[0] + sc[2]        
        sc_lst_n.append((x0, y0, x1, y1, sc[4]))
    return numpy.array(sc_lst_n)

def inria2sc(sc_lst):
    """
    take: (left, top, right, bottom, score)
    give: (i, j, half_height, half_width, score)
    """
    sc_lst_n = list()
    for sc in sc_lst:
        i = (sc[3] + sc[1])/2
        j = (sc[2] + sc[0])/2
        half_h = (sc[3] - sc[1])/2
        half_w = (sc[2] - sc[0])/2
        sc_lst_n.append((i ,j, half_h, half_w, sc[4]))
    return sc_lst_n


def compute_overlap(box0 , box1):
    """
    Take: boxes in the format of (l, t, r, b)

    Give: overlap area of two boxes
    """
    l0, t0, r0, b0 = box0[0], box0[1], box0[2], box0[3]
    l1, t1, r1, b1 = box1[0], box1[1], box1[2], box1[3]
    
    area0 = (r0 - l0 + 1) * (b0 - t0 + 1)
    area1 = (r1 - l1 + 1) * (b1 - t1 + 1)
    
    xx1 = max(l0, l1)
    yy1 = max(t0, t1)
    xx2 = min(r0, r1)
    yy2 = min(b0, b1)
    
    # compute the width and height of the bounding box
    w = max(0, xx2 - xx1 + 1)
    h = max(0, yy2 - yy1 + 1)
    
    overlap = float(w * h) / (area0+area1-w*h)
   
    return overlap
