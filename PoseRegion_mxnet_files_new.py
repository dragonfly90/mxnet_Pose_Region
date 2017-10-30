from mxnet.gluon import nn
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxPrior
import mxnet.gluon as gluon
import matplotlib.pyplot as plt
import os
import mxnet as mx
import numpy as np
import copy
import re
import json
import cv2 as cv

import scipy
import PIL.Image
import math
import time
from PIL import Image, ImageDraw

from config.config import config
from collections import namedtuple

from cython.heatmap import putGaussianMaps
from cython.pafmap import putVecMaps

import numpy as np
from bbox_transform import *
from mxnet import autograd as ag
## define anchor

n = 46
# shape: batch x channel x height x weight
x = mx.nd.random_uniform(shape=(1, 3, n, n))
y = MultiBoxPrior(x, sizes=[.5], ratios=[1])

# the first anchor box generated for pixel at (20,20)
# its format is (x_min, y_min, x_max, y_max)
boxes = y.reshape((n, n, -1, 4))
print('The first anchor box at row 21, column 21:', boxes[20, 20, 0, :])

## author: Liang Dong
## Generate heat map and part affinity map

Point = namedtuple('Point', 'x y')

crop_size_x = 368
crop_size_y = 368
center_perterb_max = 40
scale_prob = 1
scale_min = 0.5
scale_max = 1.1
target_dist = 0.6

numofparts = 18
numoflinks = 19

with open('pose_io/data.json', 'r') as f:
    datas = json.load(f)
keyss = datas.keys()

class DataBatchweight(object):
    def __init__(self, data, heatmaplabel, partaffinityglabel, heatweight, vecweight,
                 loclabel, locweight, pad=0):
        self.data = [data]
        self.label = [heatmaplabel, partaffinityglabel, 
                      heatweight, vecweight, loclabel, locweight]
        self.pad = pad

class cocoIterweightBatch:
    def __init__(self, datajson,
                 data_names, data_shapes, label_names,
                 label_shapes, batch_size = 1):

        self._data_shapes = data_shapes
        self._label_shapes = label_shapes
        self._provide_data = zip([data_names], [data_shapes])
        self._provide_label = zip(label_names, label_shapes)
        self._batch_size = batch_size

        with open(datajson, 'r') as f:
            data = json.load(f)

        self.num_batches = len(data)/5*5

        self.data = data
        
        self.cur_batch = 0

        self.keys = data.keys()

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch < self.num_batches:
            
            transposeImage_batch = []
            heatmap_batch = []
            pagmap_batch = []
            heatweight_batch = []
            vecweight_batch = []
            loclabel_batch = []
            locweight_batch = []
            
            for i in range(batch_size):
                if self.cur_batch >= 45174:
                    break
                '''
                image, mask, heatmap, pagmap, boxtarget, boxmask= getImageandLabel(
                    self.data[self.keys[self.cur_batch]])
                '''
                dirpath = 'traindata/'
                image = cv.imread(dirpath + str(self.keys[self.cur_batch]) + '_image.jpg')
                mask = np.load(dirpath + str(self.keys[self.cur_batch]) + '_mask.npy')
                heatmap = np.load(dirpath + str(self.keys[self.cur_batch]) + '_heat.npy')
                pagmap = np.load(dirpath + str(self.keys[self.cur_batch]) + '_pag.npy')
                boxtarget = np.load(dirpath + str(self.keys[self.cur_batch]) + '_boxtarget.npy')
                boxmask = np.load(dirpath + str(self.keys[self.cur_batch]) + '_boxmask.npy')
                #print len(heatmap)
                #print len(pagmap)
                
                maskscale = mask[0:368:8, 0:368:8, 0]
                
                heatweight = np.ones((19,46,46))
                vecweight = np.ones((38,46,46))
                loclabel = boxtarget
                locweight = np.ones((4,46,46))
               
                for i in range(4):
                    locweight[i,:,:] = boxmask
                    
                for i in range(19):
                    heatweight[i,:,:] = maskscale

                for i in range(38):
                    vecweight[i,:,:] = maskscale
                
                transposeImage = np.transpose(np.float32(image), (2,0,1))/256 - 0.5
            
                self.cur_batch += 1
                
                transposeImage_batch.append(transposeImage)
                heatmap_batch.append(heatmap)
                pagmap_batch.append(pagmap)
                heatweight_batch.append(heatweight)
                vecweight_batch.append(vecweight)
                loclabel_batch.append(loclabel)
                locweight_batch.append(locweight)
                
            return DataBatchweight(
                mx.nd.array(transposeImage_batch),
                mx.nd.array(heatmap_batch),
                mx.nd.array(pagmap_batch),
                mx.nd.array(heatweight_batch),
                mx.nd.array(vecweight_batch),
                mx.nd.array(loclabel_batch),
                mx.nd.array(locweight_batch))
                
        else:
            raise StopIteration

import mxnet.gluon as gluon

def PoseModel():
   
    data = mx.symbol.Variable(name='data')
    
    heatmaplabel = mx.sym.Variable("heatmaplabel")
    partaffinityglabel = mx.sym.Variable('partaffinityglabel')
    heatweight = mx.sym.Variable('heatweight')    
    vecweight = mx.sym.Variable('vecweight')
    loclabel = mx.sym.Variable('loclabel')
    locweight = mx.sym.Variable('locweight')
    
    conv1_1 = mx.symbol.Convolution(name='conv1_1', data=data , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu1_1 = mx.symbol.Activation(name='relu1_1', data=conv1_1 , act_type='relu')
    conv1_2 = mx.symbol.Convolution(name='conv1_2', data=relu1_1 , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu1_2 = mx.symbol.Activation(name='relu1_2', data=conv1_2 , act_type='relu')
    pool1_stage1 = mx.symbol.Pooling(name='pool1_stage1', data=relu1_2 , pooling_convention='full', pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
    conv2_1 = mx.symbol.Convolution(name='conv2_1', data=pool1_stage1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu2_1 = mx.symbol.Activation(name='relu2_1', data=conv2_1 , act_type='relu')
    conv2_2 = mx.symbol.Convolution(name='conv2_2', data=relu2_1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu2_2 = mx.symbol.Activation(name='relu2_2', data=conv2_2 , act_type='relu')
    pool2_stage1 = mx.symbol.Pooling(name='pool2_stage1', data=relu2_2 , pooling_convention='full', pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
    conv3_1 = mx.symbol.Convolution(name='conv3_1', data=pool2_stage1 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu3_1 = mx.symbol.Activation(name='relu3_1', data=conv3_1 , act_type='relu')
    conv3_2 = mx.symbol.Convolution(name='conv3_2', data=relu3_1 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu3_2 = mx.symbol.Activation(name='relu3_2', data=conv3_2 , act_type='relu')
    conv3_3 = mx.symbol.Convolution(name='conv3_3', data=relu3_2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu3_3 = mx.symbol.Activation(name='relu3_3', data=conv3_3 , act_type='relu')
    conv3_4 = mx.symbol.Convolution(name='conv3_4', data=relu3_3 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu3_4 = mx.symbol.Activation(name='relu3_4', data=conv3_4 , act_type='relu')
    pool3_stage1 = mx.symbol.Pooling(name='pool3_stage1', data=relu3_4 , pooling_convention='full', pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
    conv4_1 = mx.symbol.Convolution(name='conv4_1', data=pool3_stage1 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu4_1 = mx.symbol.Activation(name='relu4_1', data=conv4_1 , act_type='relu')
    conv4_2 = mx.symbol.Convolution(name='conv4_2', data=relu4_1 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu4_2 = mx.symbol.Activation(name='relu4_2', data=conv4_2 , act_type='relu')
    conv4_3_CPM = mx.symbol.Convolution(name='conv4_3_CPM', data=relu4_2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu4_3_CPM = mx.symbol.Activation(name='relu4_3_CPM', data=conv4_3_CPM , act_type='relu')
    conv4_4_CPM = mx.symbol.Convolution(name='conv4_4_CPM', data=relu4_3_CPM , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu4_4_CPM = mx.symbol.Activation(name='relu4_4_CPM', data=conv4_4_CPM , act_type='relu')
    
    conv5_1_CPM_L1 = mx.symbol.Convolution(name='conv5_1_CPM_L1', data=relu4_4_CPM , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu5_1_CPM_L1 = mx.symbol.Activation(name='relu5_1_CPM_L1', data=conv5_1_CPM_L1 , act_type='relu')
    conv5_1_CPM_L2 = mx.symbol.Convolution(name='conv5_1_CPM_L2', data=relu4_4_CPM , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu5_1_CPM_L2 = mx.symbol.Activation(name='relu5_1_CPM_L2', data=conv5_1_CPM_L2 , act_type='relu')
    conv5_2_CPM_L1 = mx.symbol.Convolution(name='conv5_2_CPM_L1', data=relu5_1_CPM_L1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu5_2_CPM_L1 = mx.symbol.Activation(name='relu5_2_CPM_L1', data=conv5_2_CPM_L1 , act_type='relu')
    conv5_2_CPM_L2 = mx.symbol.Convolution(name='conv5_2_CPM_L2', data=relu5_1_CPM_L2 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu5_2_CPM_L2 = mx.symbol.Activation(name='relu5_2_CPM_L2', data=conv5_2_CPM_L2 , act_type='relu')
    conv5_3_CPM_L1 = mx.symbol.Convolution(name='conv5_3_CPM_L1', data=relu5_2_CPM_L1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu5_3_CPM_L1 = mx.symbol.Activation(name='relu5_3_CPM_L1', data=conv5_3_CPM_L1 , act_type='relu')
    conv5_3_CPM_L2 = mx.symbol.Convolution(name='conv5_3_CPM_L2', data=relu5_2_CPM_L2 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu5_3_CPM_L2 = mx.symbol.Activation(name='relu5_3_CPM_L2', data=conv5_3_CPM_L2 , act_type='relu')
    conv5_4_CPM_L1 = mx.symbol.Convolution(name='conv5_4_CPM_L1', data=relu5_3_CPM_L1 , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    relu5_4_CPM_L1 = mx.symbol.Activation(name='relu5_4_CPM_L1', data=conv5_4_CPM_L1 , act_type='relu')
    conv5_4_CPM_L2 = mx.symbol.Convolution(name='conv5_4_CPM_L2', data=relu5_3_CPM_L2 , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    relu5_4_CPM_L2 = mx.symbol.Activation(name='relu5_4_CPM_L2', data=conv5_4_CPM_L2 , act_type='relu')
    conv5_5_CPM_L1 = mx.symbol.Convolution(name='conv5_5_CPM_L1', data=relu5_4_CPM_L1 , num_filter=38, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    conv5_5_CPM_L2 = mx.symbol.Convolution(name='conv5_5_CPM_L2', data=relu5_4_CPM_L2 , num_filter=19, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    

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
        
    conv5_1_CPM_Loc = mx.symbol.Convolution(name='conv5_1_CPM_Loc', data=relu4_4_CPM , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu5_1_CPM_Loc = mx.symbol.Activation(name='relu5_1_CPM_Loc', data=conv5_1_CPM_Loc, act_type='relu')
    conv5_2_CPM_Loc = mx.symbol.Convolution(name='conv5_2_CPM_Loc', data=relu5_1_CPM_Loc, num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu5_2_CPM_Loc = mx.symbol.Activation(name='relu5_2_CPM_Loc', data=conv5_2_CPM_Loc, act_type='relu')
    conv5_3_CPM_Loc = mx.symbol.Convolution(name='conv5_3_CPM_Loc', data=relu5_2_CPM_Loc, num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu5_3_CPM_Loc = mx.symbol.Activation(name='relu5_3_CPM_Loc', data=conv5_3_CPM_Loc, act_type='relu')
    conv5_4_CPM_Loc = mx.symbol.Convolution(name='conv5_4_CPM_Loc', data=relu5_3_CPM_Loc, num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    relu5_4_CPM_Loc = mx.symbol.Activation(name='relu5_4_CPM_Loc', data=conv5_4_CPM_Loc, act_type='relu')
    conv5_5_CPM_Loc = mx.symbol.Convolution(name='conv5_5_CPM_Loc', data=relu5_4_CPM_Loc, num_filter=4, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    
    # loclabel, locweight
    conv5_5_CPM_Loc_r = mx.symbol.Reshape(data=conv5_5_CPM_Loc, shape=(-1,), name='conv5_5_CPM_Loc_r')
    loclabelr = mx.symbol.Reshape(data=loclabel, shape=(-1, ), name='loclabelr')
    stage1_loss_Loc = mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(conv5_5_CPM_Loc_r - loclabelr))
    locweightr = mx.symbol.Reshape(data=locweight, shape=(-1,), name='locweightr')
    stage1_loss_Locw = stage1_loss_Loc*locweightr
    stage1_loss_Location  = mx.symbol.MakeLoss(stage1_loss_Locw)
    
    group = mx.symbol.Group([stage1_loss_L1, stage1_loss_L2, stage1_loss_Location])
    return group

posenet = PoseModel()
batch_size = 2
train_data = cocoIterweightBatch('pose_io/data.json',
                    'data', (batch_size, 3, 368,368),
                    ['heatmaplabel','partaffinityglabel','heatweight','vecweight',
                     'loclabel', 'locweight'],
                    [(batch_size, 19, 46, 46), (batch_size, 38, 46, 46),
                     (batch_size, 19, 46, 46), (batch_size, 38, 46, 46),
                     (batch_size, 4, 46, 46), (batch_size, 4, 46, 46)]
                    )

start_prefix =  40
class poseModule(mx.mod.Module):

    def fit(self, train_data, num_epoch, batch_size, carg_params=None, begin_epoch=0):
        
        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=[('data', (batch_size, 3, 368, 368))], label_shapes=[
        ('heatmaplabel', (batch_size, 19, 46, 46)),
        ('partaffinityglabel', (batch_size, 38, 46, 46)),
        ('heatweight', (batch_size, 19, 46, 46)),
        ('vecweight', (batch_size, 38, 46, 46)),
        ('loclabel', (batch_size, 4, 46, 46)),
        ('locweight', (batch_size, 4, 46, 46))])
   
        
        # self.init_params(mx.initializer.Xavier(rnd_type='uniform', factor_type='avg', magnitude=1))
        # mx.initializer.Uniform(scale=0.07),
        # mx.initializer.Uniform(scale=0.01)
        # mx.initializer.Xavier(rnd_type='uniform', factor_type='avg', magnitude=0.01)
        self.init_params(arg_params = carg_params, aux_params={}, allow_missing = True)
        #self.set_params(arg_params = carg_params, aux_params={},
        #                allow_missing = True)
        self.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.00004), ))
        losserror_list_heat = []
        losserror_list_paf = []
        losserror_list_loc = []
        
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            nbatch = 0
            data_iter = iter(train_data)
            end_of_batch = False
            next_data_batch = next(data_iter)
            i=0
            sumerror_heat = 0
            sumerror_paf  = 0
            sumerror_loc  = 0
            while not end_of_batch:
                data_batch = next_data_batch
                cmodel.forward(data_batch, is_train=True)       # compute predictions  
                prediction=cmodel.get_outputs()
                i=i+1
                sumloss=0
                numpixel=0
                print 'iteration: ', i
                
                
                lossiter = prediction[1].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                sumerror_heat = sumerror_heat + cls_loss
                print 'start heat: ', sumerror_heat
                    
                lossiter = prediction[0].asnumpy()
                cls_loss = np.sum(lossiter)/batch_size
                sumerror_paf = sumerror_paf + cls_loss
                print 'start paf: ', sumerror_paf
                
                lossiter = prediction[2].asnumpy()
                cls_loss = np.sum(lossiter)/batch_size
                sumerror_loc = sumerror_loc + cls_loss
                print 'start loc: ', sumerror_loc   
               
         
                cmodel.backward()   
                self.update()           
                
                '''
                if i > 10:
                    break
                '''
                try:
                    next_data_batch = next(data_iter)
                    self.prepare(next_data_batch)
                except StopIteration:
                    end_of_batch = True
                nbatch += 1
            
                    
            print '------Error-------'
           
            losserror_list_heat.append(sumerror_heat/i)
            losserror_list_paf.append(sumerror_paf/i)
            losserror_list_loc.append(sumerror_loc/i)
            
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)
            self.save_checkpoint('outputModel', start_prefix+epoch+1)
            
            train_data.reset()
        
        text_file = open("losserror_list_heat.txt", "w")
        text_file.write(' '.join([str(i) for i in losserror_list_heat]))
        text_file.close()
        text_file = open("losserror_list_paf.txt", "w")
        text_file.write(' '.join([str(i) for i in losserror_list_paf]))
        text_file.close()
        text_file = open("losserror_list_loc.txt", "w")
        text_file.write(' '.join([str(i) for i in losserror_list_loc]))
        text_file.close()
sym = ''
if config.TRAIN.head == 'vgg':
    sym = PoseModel() 

## Load parameters from vgg

warmupModel = '/data/guest_users/liangdong/liangdong/practice_demo/mxnet_CPM/model/vgg19'
testsym, arg_params, aux_params = mx.model.load_checkpoint(warmupModel, 0)
newargs = {}
for ikey in config.TRAIN.vggparams:
    newargs[ikey] = arg_params[ikey]

output_prefix = 'outputModel'
testsym, newargs, aux_params = mx.model.load_checkpoint(output_prefix, start_prefix)

cmodel = poseModule(symbol=sym, context=[mx.gpu(3), mx.gpu(2)],
                    label_names=['heatmaplabel','partaffinityglabel','heatweight','vecweight',
                     'loclabel', 'locweight'])
iteration = 10
cmodel.fit(train_data, num_epoch = iteration, batch_size = batch_size, carg_params = newargs)