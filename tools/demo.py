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


if __name__ == '__main__':
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)
    # init_frame = np.load('../chainer-pysot/init.npz')['frame']
    # init_rect = np.load('../chainer-pysot/init.npz')['init_rect']
    # second_frame = np.load('../chainer-pysot/output.npz')['second_frame']

    # tracker.init(init_frame, init_rect)
    # np.savez('../chainer-pysot/init.npz',
    #     frame=init_frame, zfs=[f.detach().cpu().numpy() for f in tracker.model.zf], init_rect=init_rect)
    # tracker.track(second_frame)
    # raise ValueError

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for frame in get_frames(args.video_name):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            np.savez('init.npz', frame=frame, init_rect=init_rect, zfs=tracker.model.zfs)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            # np.savez('output.npz', outputs=outputs, second_frame=frame)
            # raise ValueError
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1,1,2))], True, (0,255,0),3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255).astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0],bbox[1]),
                        (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,255,0),3)
            cv2.imshow(video_name, frame)
            cv2.waitKey(40)