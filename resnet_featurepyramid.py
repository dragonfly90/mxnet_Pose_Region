import cv2 as cv
import numpy as np
import scipy
import PIL.Image
import math
import json
import time
import mxnet as mx
import matplotlib
import pylab as plt

from generateLabelCPM import *

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, 
                  bn_mom=0.9, workspace=256, memonger=False):
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, 
        # a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, 
                               name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25),
                                   kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, 
                               name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), 
                                   kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, 
                               name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), 
                                   stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), 
                                          stride=stride, no_bias=True,
                                          workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, 
                               name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), 
                                   stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, 
                               name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), 
                                   stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), 
                                          stride=stride, no_bias=True, workspace=workspace, 
                                          name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def pafhead(body, level, heatmaplabel, partaffinityglabel,
            heatweight, vecweight, cname = 'pafhead'):
    conv5_1_CPM_L1 = mx.symbol.Convolution(name= cname + 'conv5_1_CPM_L1', data=body,
                                           num_filter=128, pad=(1,1), kernel=(3,3),
                                           stride=(1,1), no_bias=False)
    relu5_1_CPM_L1 = mx.symbol.Activation(name= cname + 'relu5_1_CPM_L1', data=conv5_1_CPM_L1, 
                                          act_type='relu')
    
    conv5_1_CPM_L2 = mx.symbol.Convolution(name= cname + 'conv5_1_CPM_L2', data=body , 
                                           num_filter=128, pad=(1,1), kernel=(3,3), 
                                           stride=(1,1), no_bias=False)
    relu5_1_CPM_L2 = mx.symbol.Activation(name= cname + 'relu5_1_CPM_L2', data=conv5_1_CPM_L2,
                                          act_type='relu')
    conv5_2_CPM_L1 = mx.symbol.Convolution(name= cname + 'conv5_2_CPM_L1', data=relu5_1_CPM_L1,
                                           num_filter=128, pad=(1,1), kernel=(3,3), 
                                           stride=(1,1), no_bias=False)
    relu5_2_CPM_L1 = mx.symbol.Activation(name= cname + 'relu5_2_CPM_L1', data=conv5_2_CPM_L1 , 
                                          act_type='relu')
    conv5_2_CPM_L2 = mx.symbol.Convolution(name= cname + 'conv5_2_CPM_L2', data=relu5_1_CPM_L2, 
                                           num_filter=128, pad=(1,1), kernel=(3,3), 
                                           stride=(1,1), no_bias=False)
    relu5_2_CPM_L2 = mx.symbol.Activation(name= cname + 'relu5_2_CPM_L2', data=conv5_2_CPM_L2, 
                                          act_type='relu')
    conv5_3_CPM_L1 = mx.symbol.Convolution(name= cname + 'conv5_3_CPM_L1', data=relu5_2_CPM_L1, 
                                           num_filter=128, pad=(1,1), kernel=(3,3), 
                                           stride=(1,1), no_bias=False)
    relu5_3_CPM_L1 = mx.symbol.Activation(name= cname + 'relu5_3_CPM_L1', data=conv5_3_CPM_L1 , 
                                          act_type='relu')
    conv5_3_CPM_L2 = mx.symbol.Convolution(name= cname + 'conv5_3_CPM_L2', data=relu5_2_CPM_L2 , 
                                           num_filter=128, pad=(1,1), kernel=(3,3), 
                                           stride=(1,1), no_bias=False)
    relu5_3_CPM_L2 = mx.symbol.Activation(name= cname + 'relu5_3_CPM_L2', data=conv5_3_CPM_L2 , 
                                          act_type='relu')
    conv5_4_CPM_L1 = mx.symbol.Convolution(name= cname + 'conv5_4_CPM_L1', data=relu5_3_CPM_L1 , 
                                           num_filter=512, pad=(0,0), kernel=(1,1), 
                                           stride=(1,1), no_bias=False)
    relu5_4_CPM_L1 = mx.symbol.Activation(name= cname + 'relu5_4_CPM_L1', data=conv5_4_CPM_L1 , 
                                          act_type='relu')
    conv5_4_CPM_L2 = mx.symbol.Convolution(name= cname + 'conv5_4_CPM_L2', data=relu5_3_CPM_L2 , 
                                           num_filter=512, pad=(0,0), kernel=(1,1), 
                                           stride=(1,1), no_bias=False)
    relu5_4_CPM_L2 = mx.symbol.Activation(name= cname + 'relu5_4_CPM_L2', data=conv5_4_CPM_L2 , 
                                          act_type='relu')
    conv5_5_CPM_L1 = mx.symbol.Convolution(name= cname + 'conv5_5_CPM_L1', data=relu5_4_CPM_L1 , 
                                           num_filter=38, pad=(0,0), kernel=(1,1), 
                                           stride=(1,1), no_bias=False)
    conv5_5_CPM_L2 = mx.symbol.Convolution(name= cname + 'conv5_5_CPM_L2', data=relu5_4_CPM_L2 ,
                                           num_filter=19, pad=(0,0), kernel=(1,1),
                                           stride=(1,1), no_bias=False)
    
    concat_stage2 = mx.symbol.Concat(name='concat_stage2', 
                                     *[conv5_5_CPM_L1, conv5_5_CPM_L2, body])
    
    Mconv1_stage2_L1 = mx.symbol.Convolution(name= cname + 'Mconv1_stage2_L1', data=concat_stage2 , 
                                             num_filter=128, pad=(3,3), kernel=(7,7), 
                                             stride=(1,1), no_bias=False)
    Mrelu1_stage2_L1 = mx.symbol.Activation(name= cname + 'Mrelu1_stage2_L1', data=Mconv1_stage2_L1 , 
                                            act_type='relu')
    Mconv1_stage2_L2 = mx.symbol.Convolution(name= cname + 'Mconv1_stage2_L2', data=concat_stage2 , 
                                             num_filter=128, pad=(3,3), kernel=(7,7), 
                                             stride=(1,1), no_bias=False)
    Mrelu1_stage2_L2 = mx.symbol.Activation(name= cname + 'Mrelu1_stage2_L2', data=Mconv1_stage2_L2 , 
                                            act_type='relu')
    Mconv2_stage2_L1 = mx.symbol.Convolution(name= cname + 'Mconv2_stage2_L1', data=Mrelu1_stage2_L1, 
                                             num_filter=128, pad=(3,3), kernel=(7,7), 
                                             stride=(1,1), no_bias=False)
    Mrelu2_stage2_L1 = mx.symbol.Activation(name= cname + 'Mrelu2_stage2_L1', data=Mconv2_stage2_L1 , 
                                            act_type='relu')
    Mconv2_stage2_L2 = mx.symbol.Convolution(name= cname + 'Mconv2_stage2_L2', data=Mrelu1_stage2_L2 ,
                                             num_filter=128, pad=(3,3), kernel=(7,7),
                                             stride=(1,1), no_bias=False)
    Mrelu2_stage2_L2 = mx.symbol.Activation(name= cname + 'Mrelu2_stage2_L2', data=Mconv2_stage2_L2 ,
                                            act_type='relu')
    Mconv3_stage2_L1 = mx.symbol.Convolution(name= cname + 'Mconv3_stage2_L1', data=Mrelu2_stage2_L1 ,
                                             num_filter=128, pad=(3,3), kernel=(7,7), 
                                             stride=(1,1), no_bias=False)
    Mrelu3_stage2_L1 = mx.symbol.Activation(name= cname + 'Mrelu3_stage2_L1', data=Mconv3_stage2_L1 , 
                                            act_type='relu')
    Mconv3_stage2_L2 = mx.symbol.Convolution(name= cname + 'Mconv3_stage2_L2', data=Mrelu2_stage2_L2 ,
                                             num_filter=128, pad=(3,3), kernel=(7,7), 
                                             stride=(1,1), no_bias=False)
    Mrelu3_stage2_L2 = mx.symbol.Activation(name= cname + 'Mrelu3_stage2_L2', data=Mconv3_stage2_L2 ,
                                            act_type='relu')
    Mconv4_stage2_L1 = mx.symbol.Convolution(name= cname + 'Mconv4_stage2_L1', data=Mrelu3_stage2_L1 ,
                                             num_filter=128, pad=(3,3), kernel=(7,7), 
                                             stride=(1,1), no_bias=False)
    Mrelu4_stage2_L1 = mx.symbol.Activation(name= cname + 'Mrelu4_stage2_L1', data=Mconv4_stage2_L1 ,
                                            act_type='relu')
    Mconv4_stage2_L2 = mx.symbol.Convolution(name= cname + 'Mconv4_stage2_L2', data=Mrelu3_stage2_L2 ,
                                             num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu4_stage2_L2 = mx.symbol.Activation(name= cname + 'Mrelu4_stage2_L2', data=Mconv4_stage2_L2 , act_type='relu')
    Mconv5_stage2_L1 = mx.symbol.Convolution(name= cname + 'Mconv5_stage2_L1', data=Mrelu4_stage2_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu5_stage2_L1 = mx.symbol.Activation(name= cname + 'Mrelu5_stage2_L1', data=Mconv5_stage2_L1 , act_type='relu')
    Mconv5_stage2_L2 = mx.symbol.Convolution(name= cname + 'Mconv5_stage2_L2', data=Mrelu4_stage2_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu5_stage2_L2 = mx.symbol.Activation(name= cname + 'Mrelu5_stage2_L2', data=Mconv5_stage2_L2 , act_type='relu')
    Mconv6_stage2_L1 = mx.symbol.Convolution(name= cname + 'Mconv6_stage2_L1', data=Mrelu5_stage2_L1 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    Mrelu6_stage2_L1 = mx.symbol.Activation(name= cname + 'Mrelu6_stage2_L1', data=Mconv6_stage2_L1 , act_type='relu')
    Mconv6_stage2_L2 = mx.symbol.Convolution(name= cname + 'Mconv6_stage2_L2', data=Mrelu5_stage2_L2 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    Mrelu6_stage2_L2 = mx.symbol.Activation(name= cname + 'Mrelu6_stage2_L2', data=Mconv6_stage2_L2 , act_type='relu')
    Mconv7_stage2_L1 = mx.symbol.Convolution(name= cname + 'Mconv7_stage2_L1', data=Mrelu6_stage2_L1 , num_filter=38, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    Mconv7_stage2_L2 = mx.symbol.Convolution(name= cname + 'Mconv7_stage2_L2', data=Mrelu6_stage2_L2 , num_filter=19, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    concat_stage3 = mx.symbol.Concat(name= cname + 'concat_stage3', *[Mconv7_stage2_L1, Mconv7_stage2_L2, body] )
    
    Mconv1_stage3_L1 = mx.symbol.Convolution(name= cname + 'Mconv1_stage3_L1', data=concat_stage3 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu1_stage3_L1 = mx.symbol.Activation(name= cname + 'Mrelu1_stage3_L1', data=Mconv1_stage3_L1 , act_type='relu')
    Mconv1_stage3_L2 = mx.symbol.Convolution(name= cname + 'Mconv1_stage3_L2', data=concat_stage3 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu1_stage3_L2 = mx.symbol.Activation(name= cname + 'Mrelu1_stage3_L2', data=Mconv1_stage3_L2 , act_type='relu')
    Mconv2_stage3_L1 = mx.symbol.Convolution(name= cname + 'Mconv2_stage3_L1', data=Mrelu1_stage3_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu2_stage3_L1 = mx.symbol.Activation(name= cname + 'Mrelu2_stage3_L1', data=Mconv2_stage3_L1 , act_type='relu')
    Mconv2_stage3_L2 = mx.symbol.Convolution(name= cname + 'Mconv2_stage3_L2', data=Mrelu1_stage3_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu2_stage3_L2 = mx.symbol.Activation(name= cname + 'Mrelu2_stage3_L2', data=Mconv2_stage3_L2 , act_type='relu')
    Mconv3_stage3_L1 = mx.symbol.Convolution(name= cname + 'Mconv3_stage3_L1', data=Mrelu2_stage3_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu3_stage3_L1 = mx.symbol.Activation(name= cname + 'Mrelu3_stage3_L1', data=Mconv3_stage3_L1 , act_type='relu')
    Mconv3_stage3_L2 = mx.symbol.Convolution(name= cname + 'Mconv3_stage3_L2', data=Mrelu2_stage3_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu3_stage3_L2 = mx.symbol.Activation(name= cname + 'Mrelu3_stage3_L2', data=Mconv3_stage3_L2 , act_type='relu')
    Mconv4_stage3_L1 = mx.symbol.Convolution(name= cname + 'Mconv4_stage3_L1', data=Mrelu3_stage3_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu4_stage3_L1 = mx.symbol.Activation(name= cname + 'Mrelu4_stage3_L1', data=Mconv4_stage3_L1 , act_type='relu')
    Mconv4_stage3_L2 = mx.symbol.Convolution(name= cname + 'Mconv4_stage3_L2', data=Mrelu3_stage3_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu4_stage3_L2 = mx.symbol.Activation(name= cname + 'Mrelu4_stage3_L2', data=Mconv4_stage3_L2 , act_type='relu')
    Mconv5_stage3_L1 = mx.symbol.Convolution(name= cname + 'Mconv5_stage3_L1', data=Mrelu4_stage3_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu5_stage3_L1 = mx.symbol.Activation(name= cname + 'Mrelu5_stage3_L1', data=Mconv5_stage3_L1 , act_type='relu')
    Mconv5_stage3_L2 = mx.symbol.Convolution(name= cname + 'Mconv5_stage3_L2', data=Mrelu4_stage3_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu5_stage3_L2 = mx.symbol.Activation(name= cname + 'Mrelu5_stage3_L2', data=Mconv5_stage3_L2 , act_type='relu')
    Mconv6_stage3_L1 = mx.symbol.Convolution(name= cname + 'Mconv6_stage3_L1', data=Mrelu5_stage3_L1 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    Mrelu6_stage3_L1 = mx.symbol.Activation(name= cname + 'Mrelu6_stage3_L1', data=Mconv6_stage3_L1 , act_type='relu')
    Mconv6_stage3_L2 = mx.symbol.Convolution(name= cname + 'Mconv6_stage3_L2', data=Mrelu5_stage3_L2 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    Mrelu6_stage3_L2 = mx.symbol.Activation(name= cname + 'Mrelu6_stage3_L2', data=Mconv6_stage3_L2 , act_type='relu')
    Mconv7_stage3_L1 = mx.symbol.Convolution(name= cname + 'Mconv7_stage3_L1', data=Mrelu6_stage3_L1 , num_filter=38, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    Mconv7_stage3_L2 = mx.symbol.Convolution(name= cname + 'Mconv7_stage3_L2', data=Mrelu6_stage3_L2 , num_filter=19, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    concat_stage4 = mx.symbol.Concat(name= cname + 'concat_stage4', *[Mconv7_stage3_L1, Mconv7_stage3_L2, body] )
    
    Mconv1_stage4_L1 = mx.symbol.Convolution(name= cname + 'Mconv1_stage4_L1', data=concat_stage4 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu1_stage4_L1 = mx.symbol.Activation(name= cname + 'Mrelu1_stage4_L1', data=Mconv1_stage4_L1 , act_type='relu')
    Mconv1_stage4_L2 = mx.symbol.Convolution(name= cname + 'Mconv1_stage4_L2', data=concat_stage4 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu1_stage4_L2 = mx.symbol.Activation(name= cname + 'Mrelu1_stage4_L2', data=Mconv1_stage4_L2 , act_type='relu')
    Mconv2_stage4_L1 = mx.symbol.Convolution(name= cname + 'Mconv2_stage4_L1', data=Mrelu1_stage4_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu2_stage4_L1 = mx.symbol.Activation(name= cname + 'Mrelu2_stage4_L1', data=Mconv2_stage4_L1 , act_type='relu')
    Mconv2_stage4_L2 = mx.symbol.Convolution(name= cname + 'Mconv2_stage4_L2', data=Mrelu1_stage4_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu2_stage4_L2 = mx.symbol.Activation(name= cname + 'Mrelu2_stage4_L2', data=Mconv2_stage4_L2 , act_type='relu')
    Mconv3_stage4_L1 = mx.symbol.Convolution(name= cname + 'Mconv3_stage4_L1', data=Mrelu2_stage4_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu3_stage4_L1 = mx.symbol.Activation(name= cname + 'Mrelu3_stage4_L1', data=Mconv3_stage4_L1 , act_type='relu')
    Mconv3_stage4_L2 = mx.symbol.Convolution(name= cname + 'Mconv3_stage4_L2', data=Mrelu2_stage4_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu3_stage4_L2 = mx.symbol.Activation(name= cname + 'Mrelu3_stage4_L2', data=Mconv3_stage4_L2 , act_type='relu')
    Mconv4_stage4_L1 = mx.symbol.Convolution(name= cname + 'Mconv4_stage4_L1', data=Mrelu3_stage4_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu4_stage4_L1 = mx.symbol.Activation(name= cname + 'Mrelu4_stage4_L1', data=Mconv4_stage4_L1 , act_type='relu')
    Mconv4_stage4_L2 = mx.symbol.Convolution(name= cname + 'Mconv4_stage4_L2', data=Mrelu3_stage4_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu4_stage4_L2 = mx.symbol.Activation(name= cname + 'Mrelu4_stage4_L2', data=Mconv4_stage4_L2 , act_type='relu')
    Mconv5_stage4_L1 = mx.symbol.Convolution(name= cname + 'Mconv5_stage4_L1', data=Mrelu4_stage4_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu5_stage4_L1 = mx.symbol.Activation(name= cname + 'Mrelu5_stage4_L1', data=Mconv5_stage4_L1 , act_type='relu')
    Mconv5_stage4_L2 = mx.symbol.Convolution(name= cname + 'Mconv5_stage4_L2', data=Mrelu4_stage4_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu5_stage4_L2 = mx.symbol.Activation(name= cname + 'Mrelu5_stage4_L2', data=Mconv5_stage4_L2 , act_type='relu')
    Mconv6_stage4_L1 = mx.symbol.Convolution(name= cname + 'Mconv6_stage4_L1', data=Mrelu5_stage4_L1 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    Mrelu6_stage4_L1 = mx.symbol.Activation(name= cname + 'Mrelu6_stage4_L1', data=Mconv6_stage4_L1 , act_type='relu')
    Mconv6_stage4_L2 = mx.symbol.Convolution(name= cname + 'Mconv6_stage4_L2', data=Mrelu5_stage4_L2 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    Mrelu6_stage4_L2 = mx.symbol.Activation(name= cname + 'Mrelu6_stage4_L2', data=Mconv6_stage4_L2 , act_type='relu')
    Mconv7_stage4_L1 = mx.symbol.Convolution(name= cname + 'Mconv7_stage4_L1', data=Mrelu6_stage4_L1 , num_filter=38, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    Mconv7_stage4_L2 = mx.symbol.Convolution(name= cname + 'Mconv7_stage4_L2', data=Mrelu6_stage4_L2 , num_filter=19, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    concat_stage5 = mx.symbol.Concat(name= cname + 'concat_stage5', *[Mconv7_stage4_L1, Mconv7_stage4_L2, body] )
   
    Mconv1_stage5_L1 = mx.symbol.Convolution(name= cname + 'Mconv1_stage5_L1', data=concat_stage5 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu1_stage5_L1 = mx.symbol.Activation(name= cname + 'Mrelu1_stage5_L1', data=Mconv1_stage5_L1 , act_type='relu')
    Mconv1_stage5_L2 = mx.symbol.Convolution(name= cname + 'Mconv1_stage5_L2', data=concat_stage5 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu1_stage5_L2 = mx.symbol.Activation(name= cname + 'Mrelu1_stage5_L2', data=Mconv1_stage5_L2 , act_type='relu')
    Mconv2_stage5_L1 = mx.symbol.Convolution(name= cname + 'Mconv2_stage5_L1', data=Mrelu1_stage5_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu2_stage5_L1 = mx.symbol.Activation(name= cname + 'Mrelu2_stage5_L1', data=Mconv2_stage5_L1 , act_type='relu')
    Mconv2_stage5_L2 = mx.symbol.Convolution(name= cname + 'Mconv2_stage5_L2', data=Mrelu1_stage5_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu2_stage5_L2 = mx.symbol.Activation(name= cname + 'Mrelu2_stage5_L2', data=Mconv2_stage5_L2 , act_type='relu')
    Mconv3_stage5_L1 = mx.symbol.Convolution(name= cname + 'Mconv3_stage5_L1', data=Mrelu2_stage5_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu3_stage5_L1 = mx.symbol.Activation(name= cname + 'Mrelu3_stage5_L1', data=Mconv3_stage5_L1 , act_type='relu')
    Mconv3_stage5_L2 = mx.symbol.Convolution(name= cname + 'Mconv3_stage5_L2', data=Mrelu2_stage5_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu3_stage5_L2 = mx.symbol.Activation(name= cname + 'Mrelu3_stage5_L2', data=Mconv3_stage5_L2 , act_type='relu')
    Mconv4_stage5_L1 = mx.symbol.Convolution(name= cname + 'Mconv4_stage5_L1', data=Mrelu3_stage5_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu4_stage5_L1 = mx.symbol.Activation(name= cname + 'Mrelu4_stage5_L1', data=Mconv4_stage5_L1 , act_type='relu')
    Mconv4_stage5_L2 = mx.symbol.Convolution(name= cname + 'Mconv4_stage5_L2', data=Mrelu3_stage5_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu4_stage5_L2 = mx.symbol.Activation(name= cname + 'Mrelu4_stage5_L2', data=Mconv4_stage5_L2 , act_type='relu')
    Mconv5_stage5_L1 = mx.symbol.Convolution(name= cname + 'Mconv5_stage5_L1', data=Mrelu4_stage5_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu5_stage5_L1 = mx.symbol.Activation(name= cname + 'Mrelu5_stage5_L1', data=Mconv5_stage5_L1 , act_type='relu')
    Mconv5_stage5_L2 = mx.symbol.Convolution(name= cname + 'Mconv5_stage5_L2', data=Mrelu4_stage5_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu5_stage5_L2 = mx.symbol.Activation(name= cname + 'Mrelu5_stage5_L2', data=Mconv5_stage5_L2 , act_type='relu')
    Mconv6_stage5_L1 = mx.symbol.Convolution(name= cname + 'Mconv6_stage5_L1', data=Mrelu5_stage5_L1 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    Mrelu6_stage5_L1 = mx.symbol.Activation(name= cname + 'Mrelu6_stage5_L1', data=Mconv6_stage5_L1 , act_type='relu')
    Mconv6_stage5_L2 = mx.symbol.Convolution(name= cname + 'Mconv6_stage5_L2', data=Mrelu5_stage5_L2 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    Mrelu6_stage5_L2 = mx.symbol.Activation(name= cname + 'Mrelu6_stage5_L2', data=Mconv6_stage5_L2 , act_type='relu')
    Mconv7_stage5_L1 = mx.symbol.Convolution(name= cname + 'Mconv7_stage5_L1', data=Mrelu6_stage5_L1 , num_filter=38, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    Mconv7_stage5_L2 = mx.symbol.Convolution(name= cname + 'Mconv7_stage5_L2', data=Mrelu6_stage5_L2 , num_filter=19, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    
    concat_stage6 = mx.symbol.Concat(name= cname + 'concat_stage6', *[Mconv7_stage5_L1, Mconv7_stage5_L2, body] )
    Mconv1_stage6_L1 = mx.symbol.Convolution(name= cname + 'Mconv1_stage6_L1', data=concat_stage6 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu1_stage6_L1 = mx.symbol.Activation(name= cname + 'Mrelu1_stage6_L1', data=Mconv1_stage6_L1 , act_type='relu')
    Mconv1_stage6_L2 = mx.symbol.Convolution(name= cname + 'Mconv1_stage6_L2', data=concat_stage6 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu1_stage6_L2 = mx.symbol.Activation(name= cname + 'Mrelu1_stage6_L2', data=Mconv1_stage6_L2 , act_type='relu')
    Mconv2_stage6_L1 = mx.symbol.Convolution(name= cname + 'Mconv2_stage6_L1', data=Mrelu1_stage6_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu2_stage6_L1 = mx.symbol.Activation(name= cname + 'Mrelu2_stage6_L1', data=Mconv2_stage6_L1 , act_type='relu')
    Mconv2_stage6_L2 = mx.symbol.Convolution(name= cname + 'Mconv2_stage6_L2', data=Mrelu1_stage6_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu2_stage6_L2 = mx.symbol.Activation(name= cname + 'Mrelu2_stage6_L2', data=Mconv2_stage6_L2 , act_type='relu')
    Mconv3_stage6_L1 = mx.symbol.Convolution(name= cname + 'Mconv3_stage6_L1', data=Mrelu2_stage6_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu3_stage6_L1 = mx.symbol.Activation(name= cname + 'Mrelu3_stage6_L1', data=Mconv3_stage6_L1 , act_type='relu')
    Mconv3_stage6_L2 = mx.symbol.Convolution(name= cname + 'Mconv3_stage6_L2', data=Mrelu2_stage6_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu3_stage6_L2 = mx.symbol.Activation(name= cname + 'Mrelu3_stage6_L2', data=Mconv3_stage6_L2 , act_type='relu')
    Mconv4_stage6_L1 = mx.symbol.Convolution(name= cname + 'Mconv4_stage6_L1', data=Mrelu3_stage6_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu4_stage6_L1 = mx.symbol.Activation(name= cname + 'Mrelu4_stage6_L1', data=Mconv4_stage6_L1 , act_type='relu')
    Mconv4_stage6_L2 = mx.symbol.Convolution(name= cname + 'Mconv4_stage6_L2', data=Mrelu3_stage6_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu4_stage6_L2 = mx.symbol.Activation(name= cname + 'Mrelu4_stage6_L2', data=Mconv4_stage6_L2 , act_type='relu')
    Mconv5_stage6_L1 = mx.symbol.Convolution(name= cname + 'Mconv5_stage6_L1', data=Mrelu4_stage6_L1 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu5_stage6_L1 = mx.symbol.Activation(name= cname + 'Mrelu5_stage6_L1', data=Mconv5_stage6_L1 , act_type='relu')
    Mconv5_stage6_L2 = mx.symbol.Convolution(name= cname + 'Mconv5_stage6_L2', data=Mrelu4_stage6_L2 , num_filter=128, pad=(3,3), kernel=(7,7), stride=(1,1), no_bias=False)
    Mrelu5_stage6_L2 = mx.symbol.Activation(name= cname + 'Mrelu5_stage6_L2', data=Mconv5_stage6_L2 , act_type='relu')
    Mconv6_stage6_L1 = mx.symbol.Convolution(name= cname + 'Mconv6_stage6_L1', data=Mrelu5_stage6_L1 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    Mrelu6_stage6_L1 = mx.symbol.Activation(name= cname + 'Mrelu6_stage6_L1', data=Mconv6_stage6_L1 , act_type='relu')
    Mconv6_stage6_L2 = mx.symbol.Convolution(name= cname + 'Mconv6_stage6_L2', data=Mrelu5_stage6_L2 , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    Mrelu6_stage6_L2 = mx.symbol.Activation(name= cname + 'Mrelu6_stage6_L2', data=Mconv6_stage6_L2 , act_type='relu')
    Mconv7_stage6_L1 = mx.symbol.Convolution(name= cname + 'Mconv7_stage6_L1', data=Mrelu6_stage6_L1 , num_filter=38, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    Mconv7_stage6_L2 = mx.symbol.Convolution(name= cname + 'Mconv7_stage6_L2', data=Mrelu6_stage6_L2 , num_filter=19, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)

    conv5_5_CPM_L1r = mx.symbol.Reshape(data=conv5_5_CPM_L1, shape=(-1,), name='conv5_5_CPM_L1r')
    partaffinityglabelr = mx.symbol.Reshape(data=partaffinityglabel, shape=(-1, ), name='partaffinityglabelr')
    stage1_loss_L1s = mx.symbol.square(conv5_5_CPM_L1r-partaffinityglabelr)
    vecweightw = mx.symbol.Reshape(data=vecweight, shape=(-1,), name='conv5_5_CPM_L1w')
    stage1_loss_L1w = stage1_loss_L1s*vecweightw
    stage1_loss_L1  = mx.symbol.MakeLoss(stage1_loss_L1w)
    
    conv5_5_CPM_L2r = mx.symbol.Reshape(data=conv5_5_CPM_L2, shape=(-1,), name='conv5_5_CPM_L2r')
    heatmaplabelr = mx.symbol.Reshape(data=heatmaplabel, shape=(-1, ), name='heatmaplabelr')
    stage1_loss_L2s = mx.symbol.square(conv5_5_CPM_L2r-heatmaplabelr)
    heatweightw = mx.symbol.Reshape(data=heatweight, shape=(-1,), name='conv5_5_CPM_L2w')
    stage1_loss_L2w = stage1_loss_L2s*heatweightw
    stage1_loss_L2  = mx.symbol.MakeLoss(stage1_loss_L2w)
        
    Mconv7_stage2_L1r = mx.symbol.Reshape(data=Mconv7_stage2_L1, shape=(-1,), name='Mconv7_stage2_L1')
    stage2_loss_L1s = mx.symbol.square(Mconv7_stage2_L1r - partaffinityglabelr)
    stage2_loss_L1w = stage2_loss_L1s*vecweightw
    stage2_loss_L1  = mx.symbol.MakeLoss(stage2_loss_L1w)
    
    Mconv7_stage2_L2r = mx.symbol.Reshape(data=Mconv7_stage2_L2, shape=(-1,), name='Mconv7_stage2_L2')
    stage2_loss_L2s = mx.symbol.square(Mconv7_stage2_L2r-heatmaplabelr)
    stage2_loss_L2w = stage1_loss_L2s*heatweightw
    stage2_loss_L2  = mx.symbol.MakeLoss(stage2_loss_L2w)
    
    
    Mconv7_stage3_L1r = mx.symbol.Reshape(data=Mconv7_stage3_L1, shape=(-1,), name='Mconv7_stage3_L1')
    stage3_loss_L1s = mx.symbol.square(Mconv7_stage3_L1r - partaffinityglabelr)
    stage3_loss_L1w = stage3_loss_L1s*vecweightw
    stage3_loss_L1  = mx.symbol.MakeLoss(stage3_loss_L1w)
    
    Mconv7_stage3_L2r = mx.symbol.Reshape(data=Mconv7_stage3_L2, shape=(-1,), name='Mconv7_stage3_L2')
    stage3_loss_L2s = mx.symbol.square(Mconv7_stage3_L2r-heatmaplabelr)
    stage3_loss_L2w = stage3_loss_L2s*heatweightw
    stage3_loss_L2  = mx.symbol.MakeLoss(stage3_loss_L2w)
    
    Mconv7_stage4_L1r = mx.symbol.Reshape(data=Mconv7_stage4_L1, shape=(-1,), name='Mconv7_stage4_L1')
    stage4_loss_L1s = mx.symbol.square(Mconv7_stage4_L1r - partaffinityglabelr)
    stage4_loss_L1w = stage4_loss_L1s*vecweightw
    stage4_loss_L1  = mx.symbol.MakeLoss(stage4_loss_L1w)
    
    Mconv7_stage4_L2r = mx.symbol.Reshape(data=Mconv7_stage4_L2, shape=(-1,), name='Mconv7_stage4_L2')
    stage4_loss_L2s = mx.symbol.square(Mconv7_stage4_L2r-heatmaplabelr)
    stage4_loss_L2w = stage1_loss_L2s*heatweightw
    stage4_loss_L2  = mx.symbol.MakeLoss(stage4_loss_L2w)
    
    Mconv7_stage5_L1r = mx.symbol.Reshape(data=Mconv7_stage5_L1, shape=(-1,), name='Mconv7_stage5_L1')
    stage5_loss_L1s = mx.symbol.square(Mconv7_stage5_L1r - partaffinityglabelr)
    stage5_loss_L1w = stage5_loss_L1s*vecweightw
    stage5_loss_L1  = mx.symbol.MakeLoss(stage5_loss_L1w)
    
    Mconv7_stage5_L2r = mx.symbol.Reshape(data=Mconv7_stage5_L2, shape=(-1,), name='Mconv7_stage5_L2')
    stage5_loss_L2s = mx.symbol.square(Mconv7_stage5_L2r-heatmaplabelr)
    stage5_loss_L2w = stage5_loss_L2s*heatweightw
    stage5_loss_L2  = mx.symbol.MakeLoss(stage5_loss_L2w)
    
    
    Mconv7_stage6_L1r = mx.symbol.Reshape(data=Mconv7_stage6_L1, shape=(-1,), name='Mconv7_stage3_L1')
    stage6_loss_L1s = mx.symbol.square(Mconv7_stage6_L1r - partaffinityglabelr)
    stage6_loss_L1w = stage6_loss_L1s*vecweightw
    stage6_loss_L1  = mx.symbol.MakeLoss(stage6_loss_L1w)
    
    Mconv7_stage6_L2r = mx.symbol.Reshape(data=Mconv7_stage6_L2, shape=(-1,), name='Mconv7_stage3_L2')
    stage6_loss_L2s = mx.symbol.square(Mconv7_stage6_L2r-heatmaplabelr)
    stage6_loss_L2w = stage6_loss_L2s*heatweightw
    stage6_loss_L2  = mx.symbol.MakeLoss(stage6_loss_L2w)
    
    return (stage1_loss_L1, stage1_loss_L2, stage2_loss_L1, stage2_loss_L2,
            stage3_loss_L1, stage3_loss_L2, stage4_loss_L1, stage4_loss_L2,
            stage5_loss_L1, stage5_loss_L2, stage6_loss_L1, stage6_loss_L2)

bn_mom=0.9
workspace=256
depth = 152
if depth >= 50:
    filter_list = [64, 256, 512, 1024, 2048]
else:
    filter_list = [64, 64, 128, 256, 512]

data = mx.symbol.Variable(name='data')
heatmaplabels = list()
partaffinitylabels = list()
heatweights = list()
vecweights = list()


for i in range(3):
    heatmaplabels.append(mx.sym.Variable('heatmaplabel'+str(i)))
    partaffinitylabels.append(mx.sym.Variable('partaffinityglabel'+str(i)))
    heatweights.append(mx.sym.Variable('heatweight'+str(i)))
    vecweights.append(mx.sym.Variable('vecweight'+str(i)))
    
data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), 
                          stride=(2,2), pad=(3, 3),
                          no_bias=True, name="conv0", workspace=workspace)
body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

num_stages = 3
    
memonger = False
    
if depth == 18:
    units = [2, 2, 2, 2]
    bottle_neck = False
elif depth == 152:
    units = [3, 8, 36, 3]
    bottle_neck = True

tuples = list()
for i in range(num_stages):
    body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), 
                         False, name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, 
                         workspace=workspace,
                         memonger=memonger)
    for j in range(units[i]-1):
        body = residual_unit(body, filter_list[i+1], (1,1), True, 
                             name='stage%d_unit%d' % (i + 1, j + 2),
                             bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
        
    stage1_loss_L1, stage1_loss_L2, stage2_loss_L1, stage2_loss_L2, stage3_loss_L1,stage3_loss_L2, \
    stage4_loss_L1, stage4_loss_L2, stage5_loss_L1, stage5_loss_L2, stage6_loss_L1,stage6_loss_L2  \
    = pafhead(body, i, heatmaplabels[i], partaffinitylabels[i], heatweights[i], vecweights[i], cname = 'pafhead'+str(i))
    
group = mx.symbol.Group([stage1_loss_L1, stage1_loss_L2, stage2_loss_L1, stage2_loss_L2, stage3_loss_L1,stage3_loss_L2,
    stage4_loss_L1, stage4_loss_L2, stage5_loss_L1, stage5_loss_L2, stage6_loss_L1,stage6_loss_L2])

posenetwork = resnetfrontEnd_mxnetModule(152)
