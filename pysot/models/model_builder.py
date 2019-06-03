# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        
        # build backbone
        module_name, cls_name = cfg.BACKBONE.TYPE.rsplit('.', 1)
        module = importlib.import_module(module_name)
        if cls_name.startswith('alexnet'):
            self.backbone = getattr(module, cls_name)(width_mult=cfg.BACKBONE.WIDTH_MULT)
        elif cls_name.startswith('mobile'):
            self.backbone = getattr(module, cls_name)(width_mult=cfg.BACKBONE.WIDTH_MULT,
                    used_layers=cfg.BACKBONE.LAYERS)
        else: 
            self.backbone = getattr(module, cls_name)(used_layers=cfg.BACKBONE.LAYERS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            module_name, cls_name = cfg.ADJUST.TYPE.rsplit('.', 1)
            module = importlib.import_module(module_name)
            module = getattr(module, cls_name)
            self.neck = module(cfg.BACKBONE.CHANNELS, cfg.ADJUST.ADJUST_CHANNEL)
            
        # build rpn head
        module_name, cls_name = cfg.RPN.TYPE.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        if cfg.ADJUST.ADJUST:
            channels = cfg.ADJUST.ADJUST_CHANNEL
        else:
            channels = cfg.BACKBONE.CHANNELS
        if len(channels) == 1:
            channels = channels[0]

        if cfg.RPN.WEIGHTED:
            self.rpn_head = cls(cfg.ANCHOR.ANCHOR_NUM, channels, True)
        else:
            self.rpn_head = cls(cfg.ANCHOR.ANCHOR_NUM, channels)

        # build mask head
        if cfg.MASK.MASK:
            module_name, cls_name = cfg.MASK.MASK_TYPE.rsplit('.', 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, cls_name)
            self.mask_head = cls(cfg.ADJUST.ADJUST_CHANNEL[0],
                                 cfg.ADJUST.ADJUST_CHANNEL[0],
                                 cfg.MASK.OUT_CHANNELS)
            if cfg.MASK.REFINE:
                module_name, cls_name = cfg.MASK.REFINE_TYPE.rsplit('.', 1)
                module = importlib.import_module(module_name)
                cls = getattr(module, cls_name)
                self.refine_head = cls()

    def template(self, z):
        zf = self.backbone(z)
        features = [f.cpu().detach().numpy() for f in zf]
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        neck_features = [f.cpu().detach().numpy() for f in zf]
        # np.savez(
        #     'template.npz',
        #     z=z.cpu().detach().numpy(),
        #      zf0=features[0],
        #      zf1=features[1],
        #      zf2=features[2],
        #      neck0=neck_features[0],
        #      neck1=neck_features[1],
        #      neck2=neck_features[2])
        # raise ValueError
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        features = [f.cpu().detach().numpy() for f in xf]
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        neck_features = [f.cpu().detach().numpy() for f in xf]
        zf_features = [f.cpu().detach().numpy() for f in self.zf]
        cls, loc = self.rpn_head(self.zf, xf)
        np.savez(
            'features1.npz',
            x=x.cpu().numpy(),
            conf=cls.cpu().detach().numpy(),
            loc=loc.cpu().detach().numpy(),
            zf0=zf_features[0],
            zf1=zf_features[1],
            zf2=zf_features[2],
            neck0=neck_features[0],
            neck1=neck_features[1],
            neck2=neck_features[2],
            x0=features[0], x1=features[1], x2=features[2],
            )
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }
    
    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)
