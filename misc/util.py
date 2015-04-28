from __future__ import division
import os
import sys
import operator
import numpy
import glob

def get_src_filenames(dataset_dir, phase, prefix='/home/ryan/data/caltech/data-USA/images/', suffix='jpg'):
    print "Collecting filenames of", dataset_dir
    print "File format:", suffix
    print "Phase:", phase
    
    if 'data-USA' in dataset_dir:
        if phase=='test':
            setnames = sorted(glob.glob(prefix+'set*'))[6:]
        else:
            setnames = sorted(glob.glob(prefix+'set*'))[:6]
        Vnames = []
        for setname in setnames:
            Vnames += sorted(glob.glob(setname+'/V*'))
        filenames = []
        for Vname in Vnames:
            filenames += sorted(glob.glob(Vname+'/*'+suffix))
    elif 'inria' in dataset_dir:
        filenames = sorted(glob.glob('/home/ryan/data/'+phase+'/pos/*png'))
    else:
        raise ValueError()
    return filenames
 
def break_filename(filename, dataset_dir):
    if 'data-USA' in dataset_dir:
        set_name = filename[-21:-16]
        V_name = filename[-15:-11]
        frame_name = filename[-10:-4]
        frame_num = int(frame_name[1:])
    else:
        raise ValueError()
    return set_name, V_name, frame_name, frame_num

def mkdir(tar_dir):
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)

def save_bbox(tar_dir, frame_num, sc_lst, V_name, dataset_dir):
    """
    save bboxes of an image in a Caltech bbox file

    take:
    tar_dir: saving directory
    img_idx: index number of the image
    sc_lst: (w_s, h_s, w, h, sc) // Starting from upper-left corner
    """
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)
    text_name = V_name + '.txt'

    if frame_num == 29:
        try:
            os.remove(tar_dir+'/'+text_name)
        except OSError:
            pass
        text = open(tar_dir+'/'+text_name, 'w+')
    else:
        text = open(tar_dir+'/'+text_name, 'a')

    for sc in sc_lst:
        text.write(str(frame_num)+' '+str(sc[0])+' '+str(sc[1])+' '+str(sc[2])+' '+str(sc[3])+' '+str(sc[4])+'\n')
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
