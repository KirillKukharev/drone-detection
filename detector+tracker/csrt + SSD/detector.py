from abc import ABCMeta, abstractmethod

import numpy as np
from mmdet.apis import inference_detector, init_detector


class DetectorModel:
    def __init__(self, th_conf, th_iou):
        self._th_conf, self._th_iou = th_conf, th_iou
        pass

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_bboxes(self, frame):
        pass


class DetectorSSD(DetectorModel):
    """
    Данный класс предсказывает gри помощи SSD
    """

    def __init__(self, th_conf, th_iou):
        super().__init__(th_conf, th_iou)
        self._model = init_detector("my_ssd512_full.py", "epoch_9.pth", device='cuda:0')

    def get_bboxes(self, frame):
        """
        Данный метод возвращает все рамки предсказатые детектором

        :param frame: кадр
        :return: сonfs, bboxes
        """
        bboxes_res = []
        confs = []

        results = inference_detector(self._model, frame)
        bboxes = np.vstack(results)
        labels_impt = np.where(bboxes[:, -1] > self._th_conf)[0]

        for bbox in bboxes[labels_impt]:
            left = bbox[0]
            top = bbox[1]
            right = bbox[2]
            bottom = bbox[3]

            bb = (left, top, right - left, bottom - top)
            bboxes_res.append(bb)
            confs.append(bbox[-1])

        # TODO: Добавить проверку классификатором

        return confs, bboxes_res
