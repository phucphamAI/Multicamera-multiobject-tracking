# -*- coding: utf-8 -*-

from __future__ import print_function, division

import cv2
import pickle
from tqdm import tqdm


import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from strongsort.deep.models.custom_model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_swinv2, ft_net_efficient, ft_net_NAS, ft_net_convnext, PCB, PCB_test
from strongsort.deep.models.utils import fuse_all_conv_bn
#fp16

root_p = '/media/aivn24/partition1/HungAn/Loi/Track1_People_Tracking'

try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

class ReIDModel():
    def __init__(self, config_path, model_weights):
        with open(config_path, 'r') as stream:
            self.config = yaml.load(stream, Loader=yaml.FullLoader)
        if self.config["use_swin"]:
            self.h, self.w = 224, 224
        else:
            self.h, self.w = 256, 128
        

        # set linear_num
        if self.config["use_swin"] or self.config["use_swinv2"] or self.config["use_dense"] or self.config["use_convnext"]:
            self.linear_num = 512
        elif self.config["use_efficient"]:
            self.linear_num = 1792
        elif self.config["use_NAS"]:
            self.linear_num = 4032
        else:
            self.linear_num = 2048   

        # load model 
        if self.config["use_dense"]:
            model_structure = ft_net_dense(self.config["nclasses"], stride = self.config["stride"], linear_num=self.linear_num)
        elif self.config["use_NAS"]:
            model_structure = ft_net_NAS(self.config["nclasses"], linear_num=self.linear_num)
        elif self.config["use_swin"]:
            model_structure = ft_net_swin(self.config["nclasses"], linear_num=self.linear_num)
        elif self.config["use_swinv2"]:
            model_structure = ft_net_swinv2(self.config["nclasses"], (self.h, self.w),  linear_num=self.linear_num)
        elif self.config["use_convnext"]:
            model_structure = ft_net_convnext(self.config["nclasses"], linear_num=self.linear_num)
        elif self.config["use_efficient"]:
            model_structure = ft_net_efficient(self.config["nclasses"], linear_num=self.linear_num)
        elif self.config["use_hr"]:
            model_structure = ft_net_hr(self.config["nclasses"], linear_num=self.linear_num)
        else:
            model_structure = ft_net(self.config["nclasses"], stride = self.config["stride"], ibn = self.config["ibn"], linear_num=self.linear_num)

        if self.config["PCB"]:
            model_structure = PCB(self.config["nclasses"])

        self.model = self.load_network(model_structure, model_weights)

        if self.config["PCB"]:
            #if opt.fp16:
            #    model = PCB_test(model[1])
            #else:
            self.model = PCB_test(self.model)
        else:
            #if opt.fp16:
                #model[1].model.fc = nn.Sequential()
                #model[1].classifier = nn.Sequential()
            #else:
            self.model.classifier.classifier = nn.Sequential()

        # gpu 
        self.use_gpu = torch.cuda.is_available() 
        self.model = self.model.eval()
        if self.use_gpu:
            self.model = self.model.cuda()

        # Here I fuse conv and bn for faster inference, and it does not work for transformers. Comment out this following line if you do not want to fuse conv&bn.
        #  self.model = fuse_all_conv_bn(self.model)

        if self.use_gpu:
            dummy_forward_input = torch.rand(self.config["batchsize"], 3, self.h, self.w).cuda()
            self.model = torch.jit.trace(self.model, dummy_forward_input)

        self.data_transforms = transforms.Compose([
                        transforms.Resize((self.h, self.w), interpolation=3),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

        if self.config["PCB"]:
            self.data_transforms = transforms.Compose([
                            transforms.Resize((384,192), interpolation=3),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                        ])
            self.h, self.w = 384, 192
           

        # multiple_scale: e.g. 1 1,1.1  1,1.1,1.2
        str_ms = ['1']
        self.ms = []
        for s in str_ms:
            s_f = float(s)
            self.ms.append(math.sqrt(s_f))

    def load_network(self, network, model_weights):
        # save_path = os.path.join(root_p,'net_1.pth')
        network.load_state_dict(torch.load(model_weights))

        return network

    def fliplr(self, img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)

        return img_flip
    
    def xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        return x1, y1, x2, y2
    
    def image_feature(self, img):
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        img = np.transpose(img, (2, 0, 1))

        img = torch.from_numpy(img).unsqueeze(0)
        n, c, h, w = img.size()

        ff = torch.FloatTensor(n, self.linear_num).zero_().cuda()
        if self.config["PCB"]:
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

        for i in range(2):
            if(i==1):
                img = self.fliplr(img)
            input_img = Variable(img.cuda())
            for scale in self.ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                
                input_img = Variable(img.type(torch.FloatTensor).cuda())
                outputs = self.model(input_img) 
                ff += outputs
            
        # normalize feature
        if self.config["PCB"]:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        return ff.cpu().detach().numpy()
                


        
