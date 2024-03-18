import cv2
from imutils.video import FPS


class Tracker_post:
    """
    Данный класс релизует трекер.
    """
    OPENCV_OBJECT_TRACKERS = {"csrt": cv2.TrackerCSRT_create,
                              "kcf": cv2.TrackerKCF_create,
                              "boosting": cv2.TrackerBoosting_create,
                              "mil": cv2.TrackerMIL_create,
                              "tld": cv2.TrackerTLD_create,
                              "medianflow": cv2.TrackerMedianFlow_create,
                              "mosse": cv2.TrackerMOSSE_create}

    def __init__(self, frame, bbox, tracker_name="csrt"):
        """
        tracker_name: Имя трекера str
            Название трекера:
                "csrt",
                "kcf",
                "boosting",
                "mil",
                "tld",
                "medianflow",
                "mosse"
        """
        self._tracker = Tracker_post.OPENCV_OBJECT_TRACKERS[tracker_name]()
        self.update_bbox(frame, bbox)
        self._fps = FPS().start()

        self.last_bbox = None
        self.track_history = []

    def update_bbox(self, frame, bbox):
        return self._tracker.init(frame, tuple(bbox))

    def get_bbox(self, frame):
        """
        Возвращение bbox для изображения.

        :param frame: Изображения для трекинга
        :return: Флаг успешного выполнения и коррдинаты рамки.
        """
        ok, bbox = self._tracker.update(frame)
        if ok:
            self.last_bbox = bbox
            return ok, bbox
        else:
            return False, self.last_bbox

    def _sum_track_results(self, n):
        return sum(map(int, self.track_history[-n:]))

    def show_track(self, n):
        return self._sum_track_results(n) / n > 0.5

    def del_tracker(self, n):
        return self._sum_track_results(n) == 0
