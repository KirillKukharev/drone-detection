from typing import List, Any
from ultralytics import YOLO

import numpy as np
import cv2
import gc

from detector import DetectorYOLO
from auxiliary_utils import get_most_similar_bbox
import torch
from datetime import datetime, date, time


class DronesDetector_post:
    """ Поиск и отображение движущихся объектов на видео
    """

    def __init__(self, tracker_name="csrt"):
        
        self.model = YOLO('yolov5.engine') # yolov8.pt / yolov9.pt
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._th_iou = 0.2
        self.confidence_threshold = 0.2


    def format(self, bbox, x_shape, y_shape):
        res = []
        if bbox:
            for b in bbox:
                p1 = (int(b[0]/640*x_shape), int(b[1]/640*y_shape))
                p2 = (int(b[0]/640*x_shape + b[2]/640*x_shape), int(b[1]/640*y_shape + b[3]/640*y_shape))
                res.append([p1, p2])
        return res if res else []

    def update(self, frame: np.array, id_frame=0, init=False):

        """
        Запуск слежения за объектами
        """
        x_shape = int(frame.shape[1])
        y_shape = int(frame.shape[0])
        frame= cv2.resize(frame , (640,640))
        bboxes_detect = None
        
        bboxes_res = []
        confs = []

        results = self.model.track(frame, conf=self.confidence_threshold,  iou=self._th_iou, persist=True, tracker='botsort.yaml')
        for result in results:
            boxes = result.boxes.xyxy.tolist()
            confidences = result.boxes.conf.tolist()
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box
                left = int(x1/640*x_shape)
                top = int(y1/640*y_shape)
                right = int(x2/640*x_shape)
                bottom = int(y2/640*y_shape)
                bb = [(left, top), (right, bottom)]
                if ((right-left)*(bottom-top))>350: #350
                     bboxes_res.append(bb)
        return {"bboxes_detect": [], "bbox_tracker": bboxes_res}