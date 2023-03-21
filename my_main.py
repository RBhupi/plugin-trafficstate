from __future__ import print_function

import numpy as np
import time
import argparse
from pathlib import Path

import cv2
import datetime
import os
import ffmpeg

import torch
import torch.nn as nn


from models.experimental import Ensemble
from models.common import Conv, DWConv
from utils.general import non_max_suppression, apply_classifier


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


class YOLOv7_Main():
    def __init__(self, args, weightfile):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model = Ensemble()
        ckpt = torch.load(weightfile, map_location=self.device)
        self.model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

        # Compatibility updates
        for m in self.model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

        self.model = self.model.half()
        self.model.eval()

        self.class_names = load_class_names(args.labels)


    def run(self, frame, args):
        sized = cv2.resize(frame, (640, 640))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        image = sized / 255.0
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).to(self.device).half()
        image = image.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(image)[0]
            pred = non_max_suppression(pred, args.det_thr, args.iou_thres, classes=args.classes, agnostic=True)

        return pred




def main(args):
    yolov7_main = YOLOv7_Main(args, args.weight)
    
    img_path = Path(args.path)

    for filename in img_path.glob('*.jpg'):
        frame = cv2.imread(str(filename))
        width  = frame.shape[1]
        height = frame.shape[0]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = yolov7_main.run(frame, args)
        results = np.asarray(results[0].cpu().detach())
        break
            


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('-path', type=str, default='.')
    parser.add_argument('-weight', type=str, default='yolov7.pt')

    # Data
    parser.add_argument('-labels', dest='labels',
                        action='store', default='coco.names', type=str,
                        help='Labels for detection')
    parser.add_argument("-detection-thres", dest='det_thr', type=float, default=0.5)
    parser.add_argument('-iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('-classes', nargs='+', type=int, help='filter by class: -class 0, or -class 0 2 3')

    return parser.parse_args()


if __name__ == '__main__':

    print(time.time())
    args = parse_args()

    main(args)
