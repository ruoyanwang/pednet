import caffe
import numpy


def net_init(src_proto, model, mean_file, device_id=1, dir='inria'):
    "Load a network"
    net = caffe.Net(src_proto, model)
    net.set_mode_gpu()
    net.set_device(device_id)
    net.set_phase_test()
    net.set_mean('data', mean_file)
    net.set_channel_swap('data', (2,1,0))
    net.set_input_scale('data', 255.0)

    return net


def fullconv_net_init(src_proto, tar_proto, model, mean_file, device_id, dataset):
    """ Transfer prototxts and initialize a network"""
    net = caffe.Net(src_proto, model)
    """
    patch = caffe.io.load_image('/home/ryan/data/caltech/data-USA/train/cropped_pos/00010.png')
    out = net.forward_all(data=numpy.asarray(\
                            [net.preprocess('data', patch)]))
    print out.keys()
    print out['pool5'].shape
    """
    params = ['fc6_c', 'fc7_c', 'fc8_c']
    # fc_params = {name: (weights, biases)}
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}


    net_full_conv = caffe.Net(tar_proto, model)
    params_full_conv = ['fc6_conv', 'fc7_conv', 'fc8_inria_conv']
    # conv_params = {name: (weights, biases)}
    conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

    for pr, pr_conv in zip(params, params_full_conv):
        conv_params[pr_conv][1][...] = fc_params[pr][1]

    for pr, pr_conv in zip(params, params_full_conv):
        out, in_, h, w = conv_params[pr_conv][0].shape
        # print out, in_, h, w
        W = fc_params[pr][0].reshape((out, in_, h, w))
        conv_params[pr_conv][0][...] = W

    # net_full_conv.save('inria/inria_fullconv.caffemodel')

    net_full_conv.set_mode_gpu()
    net_full_conv.set_device(device_id)
    net_full_conv.set_phase_test()
    net_full_conv.set_mean('data', mean_file)
    net_full_conv.set_channel_swap('data', (2,1,0))
    net_full_conv.set_input_scale('data', 255.0)

    return net_full_conv

