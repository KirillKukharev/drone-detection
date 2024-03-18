from typing import List, Any

import numpy as np
import cv2
import gc

from detector import DetectorYOLO
from auxiliary_utils import get_most_similar_bbox
from tracker import Tracker_post
import torch
from datetime import datetime, date, time


class DronesDetector_post:
    """ Поиск и отображение движущихся объектов на видео
    """

    def __init__(self, tracker_name="csrt"):
        """
        Parameters
        ----------
        stream_video: str
            Путь к файлу с видео или поток,
        tracker_name: str
            Название трекера:
                "csrt",
                "kcf",
                "boosting",
                "mil",
                "tld",
                "medianflow",
                "mosse"
        Attributes
        ----------
        _detector: Detector
            Детектор
        _capture: array
        _trackers: List(TrackerXXX)
            Трекер
        _detector: VideoCapture
            Видео,
        _current_frame: array([...], dtype=uint8)
            Текущий кадр,
        _fps_update: int
            Интервал обновления трекера,
        _count_capture: int
            Счетчик кадров,
        _th_iou: float
            Порог iou для схожести расмок трекера и дктектора,
        _remove_old_trackers_fps: int
            Интервал удаления трекеров

        """
        self._show_detection = True

        self._trackers = []
        self._tracker_name = tracker_name

        self._th_iou = 0.3
        self._th_conf = 0.3
        self._th_distance = 200
        self._detector = DetectorYOLO(self._th_conf, self._th_iou)

        # self._init_trackers()

        self._fps_update = 5
        self._remove_old_trackers_fps = 100

    def _init_trackers(self, frame) -> None:
        """
        Инциализация трекеров по первому кадру
        """
        _, bboxes = self._detector.get_bboxes(frame)
        if bboxes:
            self._add_trackers(frame, bboxes)

    def _add_trackers(self, bboxes, frame) -> None:
        # Добавление нового трекера по стартовой рамке
        for bbox in bboxes:
            self._trackers.append(Tracker_post(frame, bbox, tracker_name=self._tracker_name))

    def _update_history_and_trackers_bbox(self, frame, tracker, bboxes_detect, bbox_tracker):
        if bboxes_detect:
            # Из всех предсказанных детектором областях ищем наиболее похожий
            most_similar_bbox = get_most_similar_bbox(bboxes_detect, bbox_tracker)

            # Если iou больше выбраного порога - говорим что это наш объект и удаляем его из списка
            if most_similar_bbox['distance'] < 200:
                tracker.update_bbox(frame, most_similar_bbox['bbox'])
                bboxes_detect.remove(most_similar_bbox['bbox'])
                tracker.track_history.append(True)
            else:
                tracker.track_history.append(False)
        else:
            tracker.track_history.append(False)

        # Возвращает оставшиеся рамки
        return bboxes_detect

    def _update_trackers_bbox(self, frame):
        _, bboxes_detect = self._detector.get_bboxes(frame)
        bbox_tracker = None

        # Отрисовываем предсказания или нет
        # if self._show_detection:
        #   self._draw_bboxes(bboxes_detect)
        for tracker in self._trackers:
            ok, bbox_tracker = tracker.get_bbox(frame)
            if ok:
                bboxes_detect = self._update_history_and_trackers_bbox(frame, tracker, bboxes_detect, bbox_tracker)

        # Если после прохождения по всем трекерам остались рамки - значит это новые объекты
        if bboxes_detect:
            self._add_trackers(bboxes_detect, frame)
        return bboxes_detect, bbox_tracker

    def _draw_bboxes(self, bboxes: list, color=(255, 255, 0)) -> None:
        # Отрисовка bboxes формата xywh
        for bbox in bboxes:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(self._current_frame, p1, p2, color, 2, 1)

    def _draw_trackers(self, frame):
        bboxes_tracker = []
        for tracker in self._trackers:
            # Если трекер накопил состояние отрисовки - рисуем его
            ok, bbox = tracker.get_bbox(frame)
            if tracker.show_track(self._fps_update):
                if ok:
                    bboxes_tracker.append(bbox)
        return bboxes_tracker

    def format(self, bbox, x_shape, y_shape):
        res = []
        if bbox:
            for b in bbox:
                # p1 = (int(b[0]/640*1920), int(b[1]/640*1080))
                p1 = (int(b[0]/640*x_shape), int(b[1]/640*y_shape))
                # p2 = (int(b[0]/640*1920 + b[2]/640*1920), int(b[1]/640*1080 + b[3]/640*1080))
                p2 = (int(b[0]/640*x_shape + b[2]/640*x_shape), int(b[1]/640*y_shape + b[3]/640*y_shape))
                res.append([p1, p2])
        return res if res else []

    def _del_not_used_trackers(self) -> None:
        # Удаление трекера (Обновление списка трекеров),
        # если последние self.remove_old_trackers_fps*self._fps_update он не был найден

        update_trackers = []
        for tracker in self._trackers:
            if not tracker.del_tracker(self._remove_old_trackers_fps):
                update_trackers.append(tracker)
        self._trackers = update_trackers
        # gc.collect()
        del update_trackers

    def update(self, frame: np.array, id_frame=0, init=False):
        # print(self, frame.shape, id_frame)

        """
        Запуск слежения за объектами
        """
        x_shape = int(frame.shape[1])
        y_shape = int(frame.shape[0])
        frame= cv2.resize(frame , (640,640))
        bboxes_detect = None
        # Каждые self._fps_update обновляем трекер
        
        if id_frame % self._fps_update == 0 or init:
            bboxes_detect, bbox_tracker = self._update_trackers_bbox(frame)
            
        # Отрисовываем видимые трекеры
        bbox_tracker = self._draw_trackers(frame)

        # Чистим трекеры, которые ни за чем не следят
        if id_frame % self._remove_old_trackers_fps == 0:
            self._del_not_used_trackers()
            torch.cuda.empty_cache()

        return {"bboxes_detect": self.format(bboxes_detect, x_shape, y_shape), "bbox_tracker": self.format(bbox_tracker, x_shape, y_shape)}
