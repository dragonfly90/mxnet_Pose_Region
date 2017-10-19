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
    
def rotatePoint(R, pointDict):
    NewPoint = {'x':0, 'y':0}
    NewPoint['x'] = R[0,0]*pointDict['x']+R[0,1]*pointDict['y']+R[0,2]
    NewPoint['y'] = R[1,0]*pointDict['x']+R[1,1]*pointDict['y']+R[1,2]
    return NewPoint

def readmeta(data):
    meta = copy.deepcopy(data)
    meta['img_size'] = {'width': data['img_width'], 'height': data['img_height']}

    joint_self = data['joint_self']
    meta['joint_self'] = {'joints': list(), 'isVisible': list()}
    for i in range(len(joint_self)):
        currentdict = {'x': joint_self[i][0], 'y': joint_self[i][1]}
        meta['joint_self']['joints'].append(currentdict)

        # meta['joint_self']['isVisible'].append(joint_self[i][2])
        if joint_self[i][2] == 3:
            meta['joint_self']['isVisible'].append(3)
        elif joint_self[i][2] == 2:
            meta['joint_self']['isVisible'].append(2)
        elif joint_self[i][2] == 0:
            meta['joint_self']['isVisible'].append(0)
        else:
            meta['joint_self']['isVisible'].append(1)
            if (meta['joint_self']['joints'][i]['x'] < 0 or meta['joint_self']['joints'][i]['y'] < 0
                or meta['joint_self']['joints'][i]['x'] >= meta['img_size']['width'] or
                        meta['joint_self']['joints'][i]['y'] >= meta['img_size']['height']):
                meta['joint_self']['isVisible'][i] = 2

    for key in data['joint_others']:
        joint_other = data['joint_others'][key]
        #print joint_other
        meta['joint_others'][key] = {'joints': list(), 'isVisible': list()}

        for i in range(len(joint_self)):
            currentdict = {'x': joint_other[i][0], 'y': joint_other[i][1]}
            meta['joint_others'][key]['joints'].append(currentdict)

            # meta['joint_self']['isVisible'].append(joint_self[i][2])
            if joint_other[i][2] == 3:
                meta['joint_others'][key]['isVisible'].append(3)
            elif joint_other[i][2] == 2:
                meta['joint_others'][key]['isVisible'].append(2)
            elif joint_other[i][2] == 0:
                meta['joint_others'][key]['isVisible'].append(0)
            else:
                meta['joint_others'][key]['isVisible'].append(1)
                if (meta['joint_others'][key]['joints'][i]['x'] < 0 or meta['joint_others'][key]['joints'][i]['y'] < 0
                    or meta['joint_others'][key]['joints'][i]['x'] >= meta['img_size']['width'] or
                            meta['joint_others'][key]['joints'][i]['y'] >= meta['img_size']['height']):
                    meta['joint_others'][key]['isVisible'][i] = 2

    return meta


def TransformJointsSelf(meta):
    jo = meta['joint_self'].copy()
    newjo = {'joints': list(), 'isVisible': list()}
    COCO_to_ours_1 = [1, 6, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
    COCO_to_ours_2 = [1, 7, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]

    #COCO_to_ours_1 = list(range(1,15)) # [14, 6, 8, 10, 5, 7, 9, 12, 13, 11, 2, 1, 4, 3]
    #COCO_to_ours_2 = list(range(1,15)) # [14, 6, 8, 10, 5, 7, 9, 12, 13, 11, 2, 1, 4, 3]

    for i in range(numofparts):
        currentdict = {'x': (jo['joints'][COCO_to_ours_1[i] - 1]['x'] + jo['joints'][COCO_to_ours_2[i] - 1]['x']) * 0.5,
                       'y': (jo['joints'][COCO_to_ours_1[i] - 1]['y'] + jo['joints'][COCO_to_ours_2[i] - 1]['y']) * 0.5}
        newjo['joints'].append(currentdict)

        if (jo['isVisible'][COCO_to_ours_1[i] - 1] == 2 or jo['isVisible'][COCO_to_ours_2[i] - 1] == 2):
            newjo['isVisible'].append(2)
        elif (jo['isVisible'][COCO_to_ours_1[i] - 1] == 3 or jo['isVisible'][COCO_to_ours_2[i] - 1] == 3):
            newjo['isVisible'].append(3)
        else:
            isVisible = jo['isVisible'][COCO_to_ours_1[i] - 1] and jo['isVisible'][COCO_to_ours_2[i] - 1]
            newjo['isVisible'].append(isVisible)

    meta['joint_self'] = newjo


def TransformJointsOther(meta):
    for key in meta['joint_others']:
        jo = meta['joint_others'][key].copy()

        newjo = {'joints': list(), 'isVisible': list()}
        COCO_to_ours_1 = [1, 6, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
        COCO_to_ours_2 = [1, 7, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
        
        #COCO_to_ours_1 = list(range(1, 15)) #[14, 6, 8, 10, 5, 7, 9, 12, 13, 11, 2, 1, 4, 3]
        #COCO_to_ours_2 = list(range(1, 15)) # [14, 6, 8, 10, 5, 7, 9, 12, 13, 11, 2, 1, 4, 3]

        for i in range(numofparts):
            currentdict = {
                'x': (jo['joints'][COCO_to_ours_1[i] - 1]['x'] + jo['joints'][COCO_to_ours_2[i] - 1]['x']) * 0.5,
                'y': (jo['joints'][COCO_to_ours_1[i] - 1]['y'] + jo['joints'][COCO_to_ours_2[i] - 1]['y']) * 0.5}
            newjo['joints'].append(currentdict)

            if (jo['isVisible'][COCO_to_ours_1[i] - 1] == 2 or jo['isVisible'][COCO_to_ours_2[i] - 1] == 2):
                newjo['isVisible'].append(2)
            elif (jo['isVisible'][COCO_to_ours_1[i] - 1] == 3 or jo['isVisible'][COCO_to_ours_2[i] - 1] == 3):
                newjo['isVisible'].append(3)
            else:
                isVisible = jo['isVisible'][COCO_to_ours_1[i] - 1] and jo['isVisible'][COCO_to_ours_2[i] - 1]
                newjo['isVisible'].append(isVisible)

        meta['joint_others'][key] = newjo


def TransformMetaJoints(meta):
    TransformJointsSelf(meta)
    TransformJointsOther(meta)

def augmentation_scale(meta, oriImg, maskmiss):
    newmeta = copy.deepcopy(meta)
    '''
    dice2 = np.random.uniform()
    scale_multiplier = (scale_max - scale_min) * dice2 + scale_min
    if newmeta['scale_provided']>0:
        scale_abs = target_dist / newmeta['scale_provided']
        scale = scale_abs * scale_multiplier
    else:
        scale = 1
    '''
    if config.TRAIN.scale_set == False:
        scale = 1
    else:
        dice2 = np.random.uniform()
        scale_multiplier = (config.TRAIN.scale_max - config.TRAIN.scale_min) * dice2 + config.TRAIN.scale_min
        scale = 368.0/oriImg.shape[0]*scale_multiplier
        
    resizeImage = cv.resize(oriImg, (0, 0), fx=scale, fy=scale)
    maskmiss_scale = cv.resize(maskmiss, (0,0), fx=scale, fy=scale)
    
    newmeta['objpos'][0] *= scale
    newmeta['objpos'][1] *= scale

    for i in range(len(meta['joint_self']['joints'])):
        newmeta['joint_self']['joints'][i]['x'] *= scale
        newmeta['joint_self']['joints'][i]['y'] *= scale

    for i in meta['joint_others']:
        for j in range(len(meta['joint_others'][i]['joints'])):
            newmeta['joint_others'][i]['joints'][j]['x'] *= scale
            newmeta['joint_others'][i]['joints'][j]['y'] *= scale

    # newmeta4['img_width'], newmeta4['img_height']
    newmeta['img_height'] = resizeImage.shape[0]
    newmeta['img_width'] = resizeImage.shape[1]
    return (newmeta, resizeImage, maskmiss_scale)


def onPlane(p, img_size):
    if (p[0] < 0 or p[1] < 0):
        return False
    if (p[0] >= img_size[0] - 1 or p[1] >= img_size[1]):
        return False
    return True

def augmentation_flip(meta, croppedImage, maskmiss):
    dice = np.random.uniform()
    newmeta = copy.deepcopy(meta)
    #print newmeta['img_width']
    
    if config.TRAIN.flip and dice > 0.5: 
        flipImage = cv.flip(croppedImage, 1)
        maskmiss_flip = cv.flip(maskmiss, 1)

        newmeta['objpos'][0] =  newmeta['img_width'] - 1 - newmeta['objpos'][0]

        for i in range(len(meta['joint_self']['joints'])):
            newmeta['joint_self']['joints'][i]['x'] = newmeta['img_width'] - 1- newmeta['joint_self']['joints'][i]['x']

        for i in meta['joint_others']:
            for j in range(len(meta['joint_others'][i]['joints'])):
                newmeta['joint_others'][i]['joints'][j]['x'] = newmeta['img_width'] - 1 - newmeta['joint_others'][i]['joints'][j]['x']
    else:
        flipImage = croppedImage.copy()
        maskmiss_flip = maskmiss.copy()
        
    return (newmeta, flipImage, maskmiss_flip)

def augmentation_rotate(meta, flipimage, maskmiss):
    newmeta = copy.deepcopy(meta)
    dice2 = np.random.uniform()
    degree = (dice2 - 0.5)*2*config.TRAIN.max_rotate_degree 
    
    #print degree
    center = (368/2, 368/2)
    
    R = cv.getRotationMatrix2D(center, degree, 1.0)
    
    rotatedImage = cv.warpAffine(flipimage, R, (368,368))
    maskmiss_rotated = cv.warpAffine(maskmiss, R, (368,368))
    
    for i in range(len(meta['joint_self']['joints'])):
        newmeta['joint_self']['joints'][i] = rotatePoint(R, newmeta['joint_self']['joints'][i])

    for i in meta['joint_others']:
        for j in range(len(meta['joint_others'][i]['joints'])):
            newmeta['joint_others'][i]['joints'][j] = rotatePoint(R, newmeta['joint_others'][i]['joints'][j])
    
    return (newmeta, rotatedImage, maskmiss_rotated)

def augmentation_croppad(meta, oriImg, maskmiss):
    # dice_x = 0.5
    # dice_y = 0.5
    dice_x = np.random.uniform()
    dice_y = np.random.uniform()
    #float dice_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    #float dice_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    crop_x = config.TRAIN.crop_size_x
    crop_y = config.TRAIN.crop_size_y

    x_offset = int((dice_x - 0.5) * 2 * config.TRAIN.center_perterb_max)
    y_offset = int((dice_y - 0.5) * 2 * config.TRAIN.center_perterb_max)

    #print x_offset, y_offset
    newmeta2 = copy.deepcopy(meta)

    center = [x_offset + meta['objpos'][0], y_offset + meta['objpos'][1]]

    offset_left = -int(center[0] - crop_x / 2)
    offset_up = -int(center[1] - crop_y / 2)

    img_dst = np.full((crop_y, crop_x, 3), 128, dtype=np.uint8)
    maskmiss_cropped = np.full((crop_y, crop_x, 3), False, dtype=np.uint8)
    for i in range(crop_y):
        for j in range(crop_x):
            coord_x_on_img = int(center[0] - crop_x / 2 + j)
            coord_y_on_img = int(center[1] - crop_y / 2 + i)
            # print coord_x_on_img, coord_y_on_img
            if (onPlane([coord_x_on_img, coord_y_on_img], [oriImg.shape[1], oriImg.shape[0]])):
                img_dst[i, j, :] = oriImg[coord_y_on_img, coord_x_on_img, :]
                maskmiss_cropped[i, j] = maskmiss[coord_y_on_img, coord_x_on_img]
                
    newmeta2['objpos'][0] += offset_left
    newmeta2['objpos'][1] += offset_up

    for i in range(len(meta['joint_self']['joints'])):
        newmeta2['joint_self']['joints'][i]['x'] += offset_left
        newmeta2['joint_self']['joints'][i]['y'] += offset_up

    for i in meta['joint_others']:
        for j in range(len(meta['joint_others'][i]['joints'])):
            newmeta2['joint_others'][i]['joints'][j]['x'] += offset_left
            newmeta2['joint_others'][i]['joints'][j]['y'] += offset_up

    newmeta2['img_height'] = 368
    newmeta2['img_width'] = 368
    return (newmeta2, img_dst, maskmiss_cropped)

def generateLabelMap(img_aug, meta):
    thre = 0.5
    crop_size_width = 368
    crop_size_height = 368

    augmentcols = 368
    augmentrows = 368
    stride = 8
    grid_x = augmentcols / stride
    grid_y = augmentrows / stride
    sigma = 4.0
    #sigma = 10.0
    #sigma = 26.0
    
    heat_map = list()
    for i in range(numofparts+1):
        heat_map.append(np.zeros((crop_size_width / stride, crop_size_height / stride)))

    for i in range(numofparts):
        if (meta['joint_self']['isVisible'][i] <= 1):
            putGaussianMaps(heat_map[i], 368, 368, 
                            meta['joint_self']['joints'][i]['x'], meta['joint_self']['joints'][i]['y'],
                            stride, grid_x, grid_y, sigma)

        for j in meta['joint_others']:
            if (meta['joint_others'][j]['isVisible'][i] <= 1):
                putGaussianMaps(heat_map[i], 368, 368, 
                                meta['joint_others'][j]['joints'][i]['x'], 
                                meta['joint_others'][j]['joints'][i]['y'],
                                stride, grid_x, grid_y, sigma)
       
    ### put background channel
    #heat_map[numofparts] = heat_map[0]
    
    for g_y in range(grid_y):
        for g_x in range(grid_x):
            maximum=0
            for i in range(numofparts):
                if maximum<heat_map[i][g_y, g_x]:
                    maximum = heat_map[i][g_y, g_x]
            heat_map[numofparts][g_y,g_x]=max(1.0-maximum,0.0)
   
    
    mid_1 = [2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16]
    mid_2 = [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
    
    #mid_1 = [13,  1,  4, 1, 2, 4, 5, 1, 7, 8,  4, 10, 11]
    #mid_2 = [14, 14, 14, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]
    thre = 1

    pag_map = list()
    for i in range(numoflinks*2):
        pag_map.append(np.zeros((46, 46)))

    for i in range(numoflinks):
        count = np.zeros((46, 46))
        jo = meta['joint_self']

        if (jo['isVisible'][mid_1[i] - 1] <= 1 and jo['isVisible'][mid_2[i] - 1] <= 1):
            putVecMaps(pag_map[2 * i], pag_map[2 * i + 1], count,
                       jo['joints'][mid_1[i] - 1]['x'], jo['joints'][mid_1[i] - 1]['y'], 
                       jo['joints'][mid_2[i] - 1]['x'], jo['joints'][mid_2[i] - 1]['y'],
                       stride, 46, 46, sigma, thre)


        for j in meta['joint_others']:
            jo = meta['joint_others'][j]
            if (jo['isVisible'][mid_1[i] - 1] <= 1 and jo['isVisible'][mid_2[i] - 1] <= 1):
                putVecMaps(pag_map[2 * i], pag_map[2 * i + 1], count,
                           jo['joints'][mid_1[i] - 1]['x'], jo['joints'][mid_1[i] - 1]['y'],
                           jo['joints'][mid_2[i] - 1]['x'], jo['joints'][mid_2[i] - 1]['y'],
                           stride, 46, 46, sigma, thre)
                
    return (heat_map, pag_map)

def getMask(meta):
    nx, ny = meta['img_width'], meta['img_height']
    maskall = np.zeros((ny, nx))

    try: 
        if(len(meta['segmentations']) > 0):
            for i in range(len(meta['segmentations'])): 
                seg = meta['segmentations'][i]
                if len(seg) > 0:
                    nlen = len(seg[0])
                    if nlen > 5:
                        poly = zip(seg[0][0:nlen+2:2], seg[0][1:nlen+1:2])
                        img = Image.new("L", [nx, ny], 0)
                        ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
                        mask = np.array(img)
                        maskall = np.logical_or(mask, maskall)
    except:
        print 'full mask'
    
    return np.logical_not(maskall)

def targetAndMask(newmeta):
    boxtarget = np.zeros((4,46,46))
    boxmask = np.zeros((46,46))
    allboxs = list()
    alltargets = list()
    allaxis = list()
    
    ## main person 
    joints = newmeta['joint_self']['joints']
    visiblelabel = newmeta['joint_self']['isVisible']
    groundtruth = [min(newmeta['bbox'][0], newmeta['bbox'][2]), min(newmeta['bbox'][1], newmeta['bbox'][3]),
                   max(newmeta['bbox'][0], newmeta['bbox'][2]), max(newmeta['bbox'][1], newmeta['bbox'][3])]
    
    for joint, visible in zip(joints, visiblelabel):
        featurex = int(round(joint['x']/8))
        featurey = int(round(joint['y']/8))
        if visible <= 1 and featurex >= 0 and featurex < 46 and featurey >= 0 and featurey < 46:
        
            anchors = boxes[int(featurex), int(featurey), :, :]
            box = anchors[0, :].asnumpy()*368
            allboxs.append(box)
            alltargets.append(groundtruth)
            allaxis.append((featurex, featurey))
            
    ## other persons
    for key, value in newmeta['joint_others'].iteritems():
        joints = value['joints']
        visiblelabel = value['isVisible']
        cbox = newmeta['bbox_other'][key]
        groundtruth = [min(cbox[0], cbox[2]), min(cbox[1], cbox[3]),
                       max(cbox[0], cbox[2]), max(cbox[1], cbox[3])]
        
        for joint, visible in zip(joints, visiblelabel):
            featurex = int(round(joint['x']/8))
            featurey = int(round(joint['y']/8))
            if visible <= 1 and featurex >= 0 and featurex < 46 and featurey >= 0 and featurey < 46:
                anchors = boxes[int(featurex), int(featurey), :, :]
                box = anchors[0, :].asnumpy()*368
                allboxs.append(box)
                alltargets.append(groundtruth)
                allaxis.append((featurex, featurey))
            
    target = bbox_transform(np.array(allboxs), np.array(alltargets))
    
    nlen = len(allaxis)
    for i in range(nlen):
        boxmask[allaxis[i][1], allaxis[i][0]] = 1
        boxtarget[:, allaxis[i][1], allaxis[i][0]] = target[i, :] 
        
    return boxtarget, boxmask

def getImageandLabel(iterjson):

    meta = readmeta(iterjson)
    
    TransformMetaJoints(meta)

    oriImg = cv.imread(meta['img_paths'])
    maskmiss = getMask(meta)
    maskmiss = maskmiss.astype(np.uint8)
    
    newmeta, resizeImage, maskmiss_scale = augmentation_scale(meta, oriImg, maskmiss)

    newmeta2, croppedImage, maskmiss_cropped = augmentation_croppad(newmeta, resizeImage,
                                                                   maskmiss_scale)
    
    newmeta3, rotatedImage, maskmiss_rotate= augmentation_rotate(newmeta2, croppedImage, 
                                                                maskmiss_cropped)
    
    newmeta4, flipImage, maskmiss_flip = augmentation_flip(newmeta3, rotatedImage, maskmiss_rotate)

    heatmap, pagmap = generateLabelMap(flipImage, newmeta4)

    boxtarget, boxmask = targetAndMask(newmeta4)
    
    return (flipImage, maskmiss_flip, heatmap, pagmap, boxtarget, boxmask)


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

        self.num_batches = len(data)/4*4

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
                image, mask, heatmap, pagmap, boxtarget, boxmask= getImageandLabel(
                    self.data[self.keys[self.cur_batch]])
                
                #print len(heatmap)
                #print len(pagmap)
                
                maskscale = mask[0:368:8, 0:368:8, 0]
                
                heatweight = np.ones((19,46,46))
                vecweight = np.ones((38,46,46))
                loclabel = boxtarget
                locweight = np.ones((4,46,46))
                '''
                loclabel = boxtarget
                locweight = np.repeat(boxmask[np.newaxis, :, :], 4, axis=0)
                heatweight = np.repeat(maskscale[np.newaxis, :, :], 19, axis=0)
                vecweight  = np.repeat(maskscale[np.newaxis, :, :], 38, axis=0)
                '''
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

def Pose():
    def vgg_block(convs, channels):
        out = gluon.nn.HybridSequential(prefix='')
        for i in range(convs):
            out.add(gluon.nn.Conv2D(channels=channels, kernel_size=3, strides=1, 
                                    padding=(1, 1), activation='relu'))
        out.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        return out

    def vgg_stack(architecture):
        out = gluon.nn.HybridSequential(prefix='')
        for (convs, channels) in architecture:
            out.add(vgg_block(convs, channels))
        return out

    architecture = ((2,64), (2,128), (4,256))
    vgg_back = gluon.nn.HybridSequential()
    with vgg_back.name_scope():
        vgg_back.add(vgg_stack(architecture))
        vgg_back.add(gluon.nn.Conv2D(channels=512, kernel_size=3, strides=1, 
                                     padding=(1, 1), activation='relu'))
        vgg_back.add(gluon.nn.Conv2D(channels=512, kernel_size=3, strides=1, 
                                     padding=(1, 1), activation='relu'))
        vgg_back.add(gluon.nn.Conv2D(channels=256, kernel_size=3, strides=1, 
                                     padding=(1, 1), activation='relu'))
        vgg_back.add(gluon.nn.Conv2D(channels=128, kernel_size=3, strides=1, 
                                     padding=(1, 1), activation='relu'))

    heatmap = gluon.nn.HybridSequential()
    heatmap.add(vgg_back)
    heatmap.add(gluon.nn.Conv2D(channels=128, kernel_size=3, strides=1, 
                                padding=(1, 1), activation='relu'))
    heatmap.add(gluon.nn.Conv2D(channels=128, kernel_size=3, strides=1, 
                                padding=(1, 1), activation='relu'))
    heatmap.add(gluon.nn.Conv2D(channels=128, kernel_size=3, strides=1, 
                                padding=(1, 1), activation='relu'))
    heatmap.add(gluon.nn.Conv2D(channels=512, kernel_size=3, strides=1, 
                                padding=(1, 1), activation='relu'))
    heatmap.add(gluon.nn.Conv2D(channels=19, kernel_size=1, strides=1, 
                                activation='relu'))

    pafmap  = gluon.nn.HybridSequential()
    pafmap.add(vgg_back)
    pafmap.add(gluon.nn.Conv2D(channels=128, kernel_size=3, strides=1, 
                               padding=(1, 1), activation='relu'))
    pafmap.add(gluon.nn.Conv2D(channels=128, kernel_size=3, strides=1, 
                               padding=(1, 1), activation='relu'))
    pafmap.add(gluon.nn.Conv2D(channels=128, kernel_size=3, strides=1, 
                               padding=(1, 1), activation='relu'))
    pafmap.add(gluon.nn.Conv2D(channels=512, kernel_size=3, strides=1, 
                               padding=(1, 1), activation='relu'))
    pafmap.add(gluon.nn.Conv2D(channels=38, kernel_size=1, strides=1, 
                               activation='relu'))

    locprediction = gluon.nn.HybridSequential()
    locprediction.add(vgg_back)
    locprediction.add(gluon.nn.Conv2D(channels=128, kernel_size=3, strides=1, 
                                      padding=(1, 1), activation='relu'))
    locprediction.add(gluon.nn.Conv2D(channels=128, kernel_size=3, strides=1, 
                                      padding=(1, 1), activation='relu'))
    locprediction.add(gluon.nn.Conv2D(channels=128, kernel_size=3, strides=1, 
                                      padding=(1, 1), activation='relu'))
    locprediction.add(gluon.nn.Conv2D(channels=512, kernel_size=3, strides=1, 
                                      padding=(1, 1), activation='relu'))
    locprediction.add(gluon.nn.Conv2D(channels=4, kernel_size=3, strides=1, 
                                      padding=(1, 1), activation='relu'))

    # Flatten and apply fullly connected layers
    return pafmap, heatmap, locprediction

class PoseNet(gluon.Block):
    def __init__(self,  **kwargs):
        super(PoseNet, self).__init__(**kwargs)
        with self.name_scope():
            self.pafmap, self.heatmap, self.locprediction = Pose()
            
    def forward(self, x):
        return self.pafmap(x), self.heatmap(x), self.locprediction(x)
    
batch_size = 10
train_data = cocoIterweightBatch('pose_io/data.json',
                    'data', (batch_size, 3, 368,368),
                    ['heatmaplabel','partaffinityglabel','heatweight','vecweight',
                     'loclabel', 'locweight'],
                    [(batch_size, 19, 46, 46), (batch_size, 38, 46, 46),
                     (batch_size, 19, 46, 46), (batch_size, 38, 46, 46),
                     (batch_size, 4, 46, 46), (batch_size, 4, 46, 46)]
                    )

class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, mask):
        loss = F.smooth_l1((output - label) * mask, scalar=1.0)
        return F.sum(loss, exclude=True)

box_loss = SmoothL1Loss()

class L2SumLoss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(L2SumLoss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, mask):
        loss = F.square((pred - label)*mask)
        return F.sum(loss, exclude=True)

l2_loss = L2SumLoss()

### Set context for training
ctx = mx.cpu()  # it may takes too long to train using CPU

net = PoseNet()
net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)

from mxnet.gluon.model_zoo import vision
vgg = vision.vgg19()
vggparameters = vgg.collect_params()

param_list = []
for i in range(12):
    param_list.append('vgg0_conv'+str(i)+'_bias')
    param_list.append('vgg0_conv'+str(i)+'_weight')

net.collect_params().reset_ctx(ctx)
for param in param_list:
    k = param.split('_')
    net.params.setattr('posenet0_hybridsequential0_'+k[1]+'_'+k[2], 
                       vggparameters.get(param))

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1, 'wd': 5e-4})

start_epoch = 0
epochs = 3


for epoch in range(start_epoch, epochs):
    # reset iterator and tick
    
    sum_loss1 = 0
    sum_loss2 = 0
    sum_loss3 = 0 

    train_data.reset()
    tic = time.time()
    # iterate through all batch
    for i, batch in enumerate(train_data):
        btic = time.time()
        # record gradients
        with ag.record():
            '''
            x = batch.data[0].as_in_context(ctx)
            y = batch.label[0].as_in_context(ctx)
            paf_predictions, heatmap_predictions, loc_predictions = net(x)
            box_target, box_mask, cls_target = training_targets(default_anchors, 
                                                                class_predictions, y)
            '''
            # losses
            x = batch.data[0].as_in_context(ctx)
            print x.shape
            heatmaplabel = batch.label[0].as_in_context(ctx) 
            partaffinityglabel = batch.label[1].as_in_context(ctx)
            heatweight = batch.label[2].as_in_context(ctx) 
            vecweight = batch.label[3].as_in_context(ctx)
            loclabel = batch.label[4].as_in_context(ctx) 
            locweight = batch.label[5].as_in_context(ctx)
            
            
            pafmap, heatmap, locprediction = net(x)
            
            loss1 = l2_loss(pafmap.reshape((-1,)), partaffinityglabel.reshape((-1,)), 
                            vecweight.reshape((-1,)))
            loss2 = l2_loss(heatmap.reshape((-1,)), heatmaplabel.reshape((-1,)), 
                            heatweight.reshape((-1,)))
            loss3 = box_loss(locprediction.reshape((-1,)), loclabel.reshape((-1,)), 
                             locweight.reshape((-1,)))
            print 'pafmap: ', loss1
            print 'heatmap: ', loss2
            print 'regression: ', loss3
            
            sum_loss1 = sum_loss1 + loss1
            sum_loss2 = sum_loss2 + loss2
            sum_loss3 = sum_loss3 + loss3
            # sum all losses
            loss = loss1 + loss2 + loss3
            # backpropagate
            loss.backward()
        # apply
        trainer.step(batch_size)
        # update metrics
        if i > 20:
            break
            
    print 'pafmap: ', sum_loss1/20
    print 'heatmap: ', sum_loss2/20
    print 'regression: ', sum_loss3/20

    # end of epoch logging

# we can save the trained parameters to disk
net.save_params('ssd_%d.params' % epochs)

