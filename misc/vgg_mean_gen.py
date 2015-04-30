## Generate mean files in .binaryprototype and .npy formats for VGG models
import sys
sys.path.insert(0, '../../python')
import caffe
from caffe.proto import caffe_pb2
import numpy

bgr_pixels = [103.939, 116.779, 123.68]
mean_npy = numpy.ones((1, 3, 82, 41))
for i in range(3):
    mean_npy[:, i, :, :] *= bgr_pixels[i]
print mean_npy
print mean_npy.shape

# save as binaryprototype
mean_blob = caffe.io.array_to_blobproto(mean_npy) 
binaryproto_file = open('./vgg_mean.binaryproto', 'wb' )
binaryproto_file.write(mean_blob.SerializeToString())
binaryproto_file.close()

# save as .npy
numpy.save('./vgg_mean.npy',  mean_npy[0])
