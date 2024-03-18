from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from datetime import datetime, date, time


class DetectorModel:
    def __init__(self, th_conf, th_iou):
        self._th_conf, self._th_iou = th_conf, th_iou
        pass

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_bboxes(self, frame):
        pass


class DetectorYOLO(DetectorModel):
    """
    Данный класс предсказывает при помощи YOLO
    """

    def __init__(self, th_conf, th_iou):
        super().__init__(th_conf, th_iou)
        self.model = self.load_model('yolov5.engine')
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.confidence_threshold = th_conf # Минимальное значение для отображения ограничительной рамки

    def load_model(self , model_name): 
        """
        Данный метод возвращает загруженные в программу веса предобученной модели yolov8

        :param model_name: пусть к весам модели
        :return: model
        """
        model = YOLO(model_name)
        return model


    def get_bboxes(self, frame):
        """
        Данный метод возвращает все рамки предсказанные детектором

        :param frame: кадр
        :return: сonfs, bboxes
        """
        # self.model.to(self.device) # Transfer a model and its associated tensors to CPU or GPU
        # frame_width = int(frame.shape[1]/self.downscale_factor)
        # frame_height = int(frame.shape[0]/self.downscale_factor)
        # frame_resized = cv2.resize(frame , (frame_width,frame_height))
        # frame_resized = cv2.resize(frame , (640,640))
        # frame_resized = frame.resize((640,640), Image.LANCZOS)
        
        y_shape = frame.shape[0]
        # y_shape = frame.size[0]
        x_shape = frame.shape[1]
        # x_shape = frame.size[1]

        bboxes_res = []
        confs = []
        #results = self.model(frame_resized)
        results = self.model(frame, self.confidence_threshold)
        #print(results)
        #labels_impt, bboxes = results.xyxyn[0][:,-1] , results.xyxyn[0][:,:-1]
        #for infer in results[0]:
        #    for bbox in infer.boxes:
        #        print(bbox)
        for result in results:
            boxes = result.boxes.xyxy.tolist()
            confidences = result.boxes.conf.tolist()
            for box, conf in zip(boxes, confidences):
                print(box)
                x1, y1, x2, y2 = box
                left = x1
                top = y1
                right = x2
                bottom = y2
                bb = (left, top, right - left, bottom - top)
                bboxes_res.append(bb)
                
                confs.append(conf)
                # TODO: Добавить проверку классификатором
        return confs, bboxes_res
