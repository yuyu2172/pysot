from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str, help='videos or image files')
args = parser.parse_args()

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


import chainer
import chainercv

# from __future__ import division

from math import ceil
import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainercv.experimental.links.model.pspnet.transforms import \
    convolution_crop
from chainercv.links import Conv2DBNActiv
# from chainercv.links.model.resnet import ResBlock
from resblock import ResBlock
from chainercv.links import PickableSequentialChain
from chainercv import transforms
from chainercv import utils


class DilatedResBlock(ResBlock):

    def __init__(self, n_layer, in_channels, mid_channels,
                 out_channels, stride, dilate=1, groups=1, initialW=None,
                 bn_kwargs={}, stride_first=False,
                 add_seblock=False):
        if stride == 1 and dilate == 1:
            residual_conv = None
        else:
            ksize = 3
            if dilate > 1:
                dd = dilate // 2
                pad = dd
            else:
                dd = 1
                pad = 0
            residual_conv = Conv2DBNActiv(
                in_channels, out_channels, ksize, stride,
                pad, dilate=dd,
                nobias=True, initialW=initialW,
                activ=None, bn_kwargs=bn_kwargs)
        super(DilatedResBlock, self).__init__(
            n_layer, in_channels, mid_channels,
            out_channels, stride, dilate, groups, initialW,
            bn_kwargs, stride_first,
            residual_conv, add_seblock
        )


class DilatedResNet(PickableSequentialChain):

    _blocks = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
    }

    def __init__(self, n_layer, pretrained_model=None,
                 initialW=None):
        n_block = self._blocks[n_layer]

        # _, path = utils.prepare_pretrained_model(
        #     {},
        #     pretrained_model,
        #     self._models[n_layer])

        bn_kwargs = {'eps': 1e-5}

        super(DilatedResNet, self).__init__()
        with self.init_scope():
            # pad is not 3
            self.conv1 = Conv2DBNActiv(
                3, 64, 7, 2, 0, initialW=initialW,
                bn_kwargs=bn_kwargs)
            # pad is 1
            self.pool1 = lambda x: F.max_pooling_2d(
                x, ksize=3, stride=2, pad=1)
            self.res2 = DilatedResBlock(
                n_block[0], 64, 64, 256, 1, 1,
                initialW=initialW, stride_first=False,
                bn_kwargs=bn_kwargs
                )
            self.res3 = DilatedResBlock(
                n_block[1], 256, 128, 512, 2, 1,
                initialW=initialW, stride_first=False,
                bn_kwargs=bn_kwargs
                )

            # 
            self.res4 = DilatedResBlock(
                n_block[2], 512, 256, 1024, 1, 2,
                initialW=initialW, stride_first=False,
                bn_kwargs=bn_kwargs,
                )
            self.res5 = DilatedResBlock(
                n_block[3], 1024, 512, 2048, 1, 4,
                initialW=initialW, stride_first=False,
                bn_kwargs=bn_kwargs
                )

        # if path:
        #     chainer.serializers.load_npz(path, self, ignore_names=None)





if __name__ == '__main__':
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    chainer_model = DilatedResNet(50)

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    # model.eval().to(device)
    model.eval()

    res = model.backbone

    def copy_conv(l, key, val):
        print(key)
        l.W.data[:] = val.cpu().numpy()

    def copy_bn(l, key, val):
        print(key)
        if key[-6:] == 'weight':
            l.gamma.data[:] = val.cpu().numpy()
        elif key[-4:] == 'bias':
            l.beta.data[:] = val.cpu().numpy()
        elif key[-12:] == 'running_mean':
            l.avg_mean.data[:] = val.cpu().numpy()
        elif key[-11:] == 'running_var':
            l.avg_var.data[:] = val.cpu().numpy()

    weight = torch.load(args.snapshot)
    start = len('backbone.')
    for key, val in weight.items():
        if 'backbone.' == key[:start]:
            key = key[start:]
            # print(key)
            if 'conv1' == key[:5]:
                copy_conv(chainer_model.conv1.conv, key, val)
            if 'bn1' == key[:3]:
                copy_bn(chainer_model.conv1.bn, key, val)
            
            if 'layer' == key[:5]:
                s_keys = key.split('.')
                m = key[5]
                res_l = getattr(chainer_model, 'res{}'.format(int(m) + 1))
                if '0' == s_keys[1]:
                    base_l = getattr(res_l, 'a')
                else:
                    base_l = getattr(res_l, 'b{}'.format(s_keys[1]))

                if 'conv' in s_keys[2]:
                    n = s_keys[2][len('conv'):]
                    conv = getattr(
                        base_l, 'conv{}'.format(n))
                    copy_conv(conv.conv, key, val)
                elif 'bn' in s_keys[2]:
                    n = s_keys[2][len('bn'):]
                    conv = getattr(
                        base_l, 'conv{}'.format(n))
                    copy_bn(conv.bn, key, val)
                elif 'downsample' == s_keys[2]:
                    if s_keys[3] == '0':
                        l = base_l.residual_conv.conv
                        copy_conv(l, key, val)
                    elif s_keys[3] == '1':
                        l = base_l.residual_conv.bn
                        copy_bn(l, key, val)
                        
    
    chainer.config.train = False

    value = np.random.uniform(size=(1, 3, 255, 255)).astype(np.float32)
    tensor = torch.FloatTensor(value)

    out = res.conv1(tensor).cpu().detach().numpy()
    c_out = chainer_model.conv1.conv(value).data
    np.testing.assert_almost_equal(c_out, out, decimal=5)

    p = res.maxpool(res.relu(res.bn1(res.conv1(tensor))))
    out = p.cpu().detach().numpy()
    chainer_model.pick = 'pool1'
    c_out = chainer_model(value).data
    np.testing.assert_almost_equal(c_out, out, decimal=5)

    out = res.layer1[0].conv1(p).cpu().detach().numpy()
    c_out = chainer_model.res2.a.conv1.conv(c_out).data
    np.testing.assert_almost_equal(c_out, out, decimal=5)
    conv_out = out
    conv_out_t = torch.FloatTensor(conv_out)

    out = res.layer1[0].bn1(conv_out_t).cpu().detach().numpy()
    c_out = chainer_model.res2.a.conv1.bn(conv_out).data
    np.testing.assert_almost_equal(c_out, out, decimal=5)

    outs = [v.cpu().detach().numpy() for v in res(tensor)]
    [print(out.shape) for out in outs]
    chainer_model.pick = ('res3', 'res4', 'res5')
    c_out = chainer_model(value)
    for i in range(3):
        np.testing.assert_almost_equal(
            c_out[i].data, outs[i], decimal=5)




    # build tracker
    # tracker = build_tracker(model)

    # first_frame = True
    # if args.video_name:
    #     video_name = args.video_name.split('/')[-1].split('.')[0]
    # else:
    #     video_name = 'webcam'
    # cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    # for frame in get_frames(args.video_name):
    #     if first_frame:
    #         try:
    #             init_rect = cv2.selectROI(video_name, frame, False, False)
    #         except:
    #             exit()
    #         tracker.init(frame, init_rect)
    #         first_frame = False
    #     else:
    #         outputs = tracker.track(frame)
    #         if 'polygon' in outputs:
    #             polygon = np.array(outputs['polygon']).astype(np.int32)
    #             cv2.polylines(frame, [polygon.reshape((-1,1,2))], True, (0,255,0),3)
    #             mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255).astype(np.uint8)
    #             mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
    #             frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
    #         else:
    #             bbox = list(map(int, outputs['bbox']))
    #             cv2.rectangle(frame, (bbox[0],bbox[1]),
    #                     (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,255,0),3)
    #         cv2.imshow(video_name, frame)
    #         cv2.waitKey(40)