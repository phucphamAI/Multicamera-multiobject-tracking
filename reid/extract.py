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
from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_swinv2, ft_net_efficient, ft_net_NAS, ft_net_convnext, PCB, PCB_test
from utils import fuse_all_conv_bn
#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='1', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50_2', type=str, help='save model path')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_efficient', action='store_true', help='use efficient-b4' )
parser.add_argument('--use_hr', action='store_true', help='use hr18 net' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ibn', action='store_true', help='use ibn.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--video_path',default='../results_OD/c001.mp4',type=str, help='video path')
parser.add_argument('--result_path',default='../results_OD/c001.txt',type=str, help='txt path exacted by yolo')
parser.add_argument('--pkl_path',default='../results_OD/c001.pkl',type=str, help='save pkl path')

opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml'
opt.fp16 = config['fp16'] 
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']
if 'use_swin' in config:
    opt.use_swin = config['use_swin']
if 'use_swinv2' in config:
    opt.use_swinv2 = config['use_swinv2']
if 'use_convnext' in config:
    opt.use_convnext = config['use_convnext']
if 'use_efficient' in config:
    opt.use_efficient = config['use_efficient']
if 'use_hr' in config:
    opt.use_hr = config['use_hr']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 

if 'ibn' in config:
    opt.ibn = config['ibn']
if 'linear_num' in config:
    opt.linear_num = config['linear_num']

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir
video_path = opt.video_path
result_path = opt.result_path   
pkl_path = opt.pkl_path
camera_id = result_path.split('/')[-1].split('.')[0]
print(f'******* Extract features of {camera_id} Camera *******')

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
if opt.use_swin:
    h, w = 224, 224
else:
    h, w = 256, 128

data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ############### Ten Crop        
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop) 
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
    h, w = 384, 192


use_gpu = torch.cuda.is_available()
######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def image_feature(model, img):
    global h, w 
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    img = np.transpose(img, (2, 0, 1))

    opt.batchsize = 1  # set batch size to 1 since we're processing a single image
    
    if opt.linear_num <= 0:
        if opt.use_swin or opt.use_swinv2 or opt.use_dense or opt.use_convnext:
            opt.linear_num = 1024
        elif opt.use_efficient:
            opt.linear_num = 1792
        elif opt.use_NAS:
            opt.linear_num = 4032
        else:
            opt.linear_num = 2048   

    img = torch.from_numpy(img).unsqueeze(0)
    n, c, h, w = img.size()

    ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()
    if opt.PCB:
        ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

    for i in range(2):
        if(i==1):
            img = fliplr(img)
        input_img = Variable(img.cuda())
        for scale in ms:
            if scale != 1:
                # bicubic is only  available in pytorch>= 1.1
                input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
            
            input_img = Variable(img.type(torch.FloatTensor).cuda())
            outputs = model(input_img) 
            ff += outputs
        
    # normalize feature
    if opt.PCB:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
    else:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

    return ff.cpu().detach().numpy()

def xywh_to_xyxy(bbox_xywh):
    x, y, w, h = bbox_xywh

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    return x1, y1, x2, y2
    
if __name__ == "__main__":
    ########## Initialize ##########
    pkl_dict = {}

    ########## Load model ##########
    # Load Collected data Trained model
    print('-------test-----------')
    if opt.use_dense:
        model_structure = ft_net_dense(opt.nclasses, stride = opt.stride, linear_num=opt.linear_num)
    elif opt.use_NAS:
        model_structure = ft_net_NAS(opt.nclasses, linear_num=opt.linear_num)
    elif opt.use_swin:
        model_structure = ft_net_swin(opt.nclasses, linear_num=opt.linear_num)
    elif opt.use_swinv2:
        model_structure = ft_net_swinv2(opt.nclasses, (h,w),  linear_num=opt.linear_num)
    elif opt.use_convnext:
        model_structure = ft_net_convnext(opt.nclasses, linear_num=opt.linear_num)
    elif opt.use_efficient:
        model_structure = ft_net_efficient(opt.nclasses, linear_num=opt.linear_num)
    elif opt.use_hr:
        model_structure = ft_net_hr(opt.nclasses, linear_num=opt.linear_num)
    else:
        model_structure = ft_net(opt.nclasses, stride = opt.stride, ibn = opt.ibn, linear_num=opt.linear_num)

    if opt.PCB:
        model_structure = PCB(opt.nclasses)

    model = load_network(model_structure)

        # Remove the final fc layer and classifier layer
    if opt.PCB:
        #if opt.fp16:
        #    model = PCB_test(model[1])
        #else:
            model = PCB_test(model)
    else:
        #if opt.fp16:
            #model[1].model.fc = nn.Sequential()
            #model[1].classifier = nn.Sequential()
        #else:
            model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()
    
    print('Here I fuse conv and bn for faster inference, and it does not work for transformers. Comment out this following line if you do not want to fuse conv&bn.')
    model = fuse_all_conv_bn(model)

    dummy_forward_input = torch.rand(opt.batchsize, 3, h, w).cuda()
    model = torch.jit.trace(model, dummy_forward_input)

    print(model)

    ########## Load Result ##########
    with open(result_path, 'r') as f:
        lines = f.readlines()

    ########## Load Videos ##########
    cap = cv2.VideoCapture(video_path)
    count = 0

    ########## Processing ##########
    for line in tqdm(lines):
        line = line.split(' ')
        frame_id, tracklet_id = line[0:2]

        bbox_xywh = [int(i) for i in line[2:-5]]

        while(True):
            if count != int(frame_id) and int(frame_id) < count:
                ret, frame = cap.read()
                count += 1
                break

            elif count != int(frame_id) and int(frame_id) > count:
                ret, frame = cap.read()
                count += 1

            elif count == int(frame_id):
                # cv2.imwrite('/media/aivn24/partition1/HungAn/Tuong/test.jpg', frame)
                # print(count)
                # print(line)
                x1, y1, x2, y2 = xywh_to_xyxy(bbox_xywh)
                img = frame[y1:y2, x1:x2]
                # cv2.imwrite('a.jpg', img)

                feats = image_feature(model, img)

                key = f"{camera_id}_{frame_id}_{tracklet_id}"
                infos = {key:feats}

                pkl_dict.update(infos)

                break

    # Save the object to a pickle file
    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_dict, f)

    cap.release()
    cv2.destroyAllWindows()
