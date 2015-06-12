#!/usr/bin/env sh
# Create the imagenet leveldb inputs
# N.B. set the path to the imagenet train + val data dirs

TOOLS=/home/ryan/caffe-0.999/build/tools
# DATA=/home/ryan/caffe-0.999/data/inria


echo "Creating leveldb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    /home/ryan/caffe-0.999/pednet/ref/ \
    /home/ryan/caffe-0.999/pednet/ref/proposal.txt \
    proposals_leveldb 0

echo "Done."
