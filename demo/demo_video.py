#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import pathlib
import numpy as np
import cv2
import math
import json
import sys

import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

sys.path.insert(0, os.path.join('..', 'main'))
sys.path.insert(0, os.path.join('..', 'data'))
sys.path.insert(0, os.path.join('..', 'common'))
from config import cfg
from model import get_pose_net
from utils.pose_utils import process_bbox
from dataset import generate_patch_image

logger = logging.getLogger('demo_video')
logger.setLevel(logging.INFO)

def draw_result(img, bbox_list, root_list, zmin, zmax):
    for bbox in bbox_list:
        x0, y0, w, h = bbox
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (0, 0, 255))

    for root_3d in root_list:
        depth = root_3d[2] / 1000 # メートル単位にする
        blue = int(256 * (depth - zmin) / (zmax - zmin))
        if blue < 0:
            blue = 0
        if blue > 255:
            blue = 255
        color = (blue, 0, 255 - blue)
        cv2.circle(img, (int(root_3d[0]), int(root_3d[1])), radius=5, color=color, thickness=-1, lineType=cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, '{0:.2f}'.format(depth), (int(root_3d[0])+3, int(root_3d[1])-3), font, 1, (255,255,255), 1, cv2.LINE_AA)
    return img

def get_detectron2_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor

def detect_people(predictor, img):
    outputs = predictor(img)
    class_array = outputs["instances"].pred_classes.to('cpu').numpy()
    bbox_array = outputs["instances"].pred_boxes.to('cpu').tensor.numpy()
    people_bbox = []
    for i in range(class_array.shape[0]):
        if class_array[i] == 0:
            left, top, right, bottom = bbox_array[i]
            box = [left, top, right - left, bottom - top]
            people_bbox.append(box)
    return people_bbox

def estimate_root(model, transform, original_img, bbox_list):
    original_img_height, original_img_width, _ = original_img.shape
    # normalized camera intrinsics
    focal = [1500, 1500] # x-axis, y-axis
    princpt = [original_img_width/2, original_img_height/2] # x-axis, y-axis
    person_num = len(bbox_list)
    root_3d_list = []
    for n in range(person_num):
        left = bbox_list[n][0]
        top = bbox_list[n][1]
        bbox = process_bbox(bbox_list[n], original_img_width, original_img_height)
        img, img2bb_trans = generate_patch_image(original_img, bbox, False, 0.0) 
        img = transform(img).to('cuda')[None,:,:,:]
        k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*focal[0]*focal[1]/(bbox[2]*bbox[3]))]).astype(np.float32)
        k_value = torch.FloatTensor([k_value]).to('cuda')[None,:]

        with torch.no_grad():
            root_3d = model(img, k_value) # x,y: pixel, z: root-relative depth (mm)

        img = img[0].to('cpu').numpy()
        root_3d = root_3d[0].to('cpu').numpy()
        root_3d[0] = root_3d[0] / cfg.output_shape[1] * bbox[2] + bbox[0]
        root_3d[1] = root_3d[1] / cfg.output_shape[1] * bbox[3] + bbox[1]
        root_3d_list.append(root_3d)
    return root_3d_list

def rootnet_test(input_video, output_video, gpu_ids, zmin, zmax):
    cfg.set_args(gpu_ids)
    cudnn.benchmark = True

    # snapshot load
    model_path = 'snapshot_18.pth.tar'
    assert os.path.exists(model_path), 'Cannot find model at ' + model_path
    model = get_pose_net(cfg, False)
    model = DataParallel(model).to('cuda')
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['network'])
    model.eval()

    dt2 = get_detectron2_predictor()

    # prepare input image
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
    cap = cv2.VideoCapture(input_video)
    if cap.isOpened() is False:
        logger.error("Error opening input video stream or file: {0}".format(args.video))
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))    

    frame = 0
    while cap.isOpened():
        ret_val, img = cap.read()
        if not ret_val:
            break
        bbox_list = detect_people(dt2, img)
        root_list = estimate_root(model, transform, img, bbox_list)
        # x, y: pixel, z: mm
        print('frame {0}: '.format(frame), end='')
        for root in root_list:
            print('({0:.1f}, {1:.1f}, {2:.0f})'.format(root[0], root[1], root[2]), end='')
        print('')
        img = draw_result(img, bbox_list, root_list, zmin, zmax)
        out.write(img)
        frame += 1

    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RootNet demo video version')
    parser.add_argument('--zmin', type=float, default=3.0)
    parser.add_argument('--zmax', type=float, default=8.0)
    parser.add_argument('--gpu_id', type=str, default=0)
    parser.add_argument('input_video', type=str)
    parser.add_argument('output_video', type=str)
    arg = parser.parse_args()
    assert arg.zmax > arg.zmin, 'zmax must be greater than zmin'
    rootnet_test(arg.input_video, arg.output_video, arg.gpu_id, arg.zmin, arg.zmax)
