import datetime
import os
import csv

from imutils.video import FPS
import imutils
import cv2

from constants import *


class Tracker:
    """ Поиск и отображение движущихся объектов на видео

    Parameters
    ----------
    path_video: str
        Путь к файлу с видео, 
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
    _tracker_name: str
        Название трекера
    _tracker: TrackerXXX
        Трекер
    _capture: VideoCapture
        Видео, 
    _current_frame: array([...], dtype=uint8)
        Текущий кадр, 
    _frame_height: int
        Высота кадра, 
    _frame_width: int
        Ширина кадра
    _box: (int, int, int, int)
        Окаймляющий прямоугольник (x, y, w, h)
    _fps: float
        Счетчик кадров в секунду
    """

    def __init__(self, path_video, tracker_name):
        self._csv_file_name = DEFAULT_CSV_NAME
        self._init_csv_file_name(path_video)
        self._writer = None
        self._tracker_name = tracker_name
        self._tracker = None
        self._capture = cv2.VideoCapture(path_video)
        _, self._current_frame = self._capture.read()
        self._amount_frame = 1
        # self._current_frame = imutils.resize(
        #     self._current_frame, width=1200
        # )
        self._frame_height, self._frame_width = self._current_frame.shape[:2]
        self._foreground_mask = None
        self._box = None
        self._fps = None

    def _init_csv_file_name(self, path_video):
        """ Инициализировать имя файла с разметкой """
        try:
            os.makedirs(DIRECTORY_SAVING)
        except OSError:
            pass
        now = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_"))
        name_video = os.path.basename(path_video).split(".")[0]
        self._csv_file_name = DIRECTORY_SAVING + now + name_video + ".csv"

    def run(self):
        """ Выполнить трекинг объектов на видео """
        background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(
            history=3, backgroundRatio=0.95
        )
        with open(self._csv_file_name, "w", newline="") as csv_file:
            fieldnames = ["frame", "x", "y", "w", "h", "logs"]
            self._writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            self._writer.writeheader()
            frame_start = 0
            frame_end = 20825
            while True:
                print(self._amount_frame)
                if self._current_frame is None:
                    break
                # self._current_frame = imutils.resize(
                #     self._current_frame, width=1200
                # )
                if self._amount_frame == frame_end:
                    break
                if (
                    frame_start <= self._amount_frame <= 5150
                    # self._amount_frame <= 5700
                    or 6025 <= self._amount_frame <= 6525
                    or 8450 <= self._amount_frame <= 8650
                    or 8775 <= self._amount_frame <= 9750
                    or 10500 <= self._amount_frame <= 11075
                    or 11800 <= self._amount_frame <= 11950
                    or 12300 <= self._amount_frame <= 13825
                    or 14250 <= self._amount_frame <= 15325
                    or 15450 <= self._amount_frame <= 15650
                    or 16475 <= self._amount_frame <= 17575
                    or 18375 <= self._amount_frame <= 18950
                    # or 10850 <= self._amount_frame
                    or 19450 <= self._amount_frame <= frame_end
                ):
                    self._foreground_mask = background_subtractor.apply(
                        self._current_frame
                    )
                    self._tracking_object()
                    cv2.namedWindow(DEFAULT_FRAME_WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.imshow(DEFAULT_FRAME_WINDOW_NAME, self._current_frame)                
                    if self._amount_frame == 1 or self._amount_frame == frame_start:                                
                        self._select_box()
                    if self._check_commands() == EXIT_SUCCESS:
                        break
                _, self._current_frame = self._capture.read()
                self._amount_frame += 1
        self._capture.release()
        cv2.destroyAllWindows()

    def _tracking_object(self):
        """ Отобразить информацию о трекинге """
        if self._box is None:
            return
        (success, self._box) = self._tracker.update(self._current_frame)
        print(f"{self._amount_frame} {success} - {self._box}")
        if not success:
            self._box = None
            return
        self._update_box_and_tracker()
        (x, y, w, h) = [int(v) for v in self._box]
        cv2.rectangle(
            self._current_frame, (x, y),
            (x + w, y + h), (0, 0, 255), 1
        )
        self._writer.writerow({
            "frame": self._amount_frame - 1,
            "x": x,
            "y": y,
            "w": w,
            "h": h
        })
        print(f"update ({x}, {y}, {w}, {h})")
        self._fps.update()
        self._fps.stop()
        self._drow_information_text(success)

    def _update_box_and_tracker(self):
        self._process_foreground_mask()
        contours, _ = cv2.findContours(
            self._foreground_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if self._is_intersecting_boxes(self._box, (x, y, w, h)):
                persent = 0.2
                square_box = self._box[2] * self._box[3]                
                if square_box * (1 - persent) < w * h < square_box * (1 + persent):
                    self._box = (x, y, w, h)
                    self._tracker = OPENCV_OBJECT_TRACKERS[self._tracker_name]()
                    self._tracker.init(self._current_frame, self._box)

    def _is_intersecting_boxes(self, first_box, second_box):
        """ Пересекаются ли области 

            Parameters
            ----------
            first_rectangle : (int, int, int, int)
                Координаты области  (x1, y1, w, h)
            second_rectangle : (int, int, int, int)
                Координаты области (x1, y1, w, h)

            Returns
            -------
            True / False - пересекаются ли области
        """
        r1_x1, r1_y1, r1_w, r1_h = first_box
        r2_x1, r2_y1, r2_w, r2_h = second_box
        if (
            r1_x1 > r2_x1 + r2_w or
            r1_x1 + r1_w < r2_x1 or
            r1_y1 > r2_y1 + r2_h or
            r1_y1 + r1_h < r2_y1
        ):
            return False
        return True

    def _process_foreground_mask(self):
        """ Обработать маску текущего кадра """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._foreground_mask = cv2.morphologyEx(
            self._foreground_mask.copy(), cv2.MORPH_CLOSE, kernel
        )
        kornel_size = (3, 3)
        sigma = 1
        self._foreground_mask = cv2.GaussianBlur(
            self._foreground_mask.copy(), kornel_size, sigma
        )

    def _drow_information_text(self, success):
        """ Отобразить информацию о трекинге

        Parameters
        ----------
        success: bool
        """
        info = self._create_information_text(success)
        for i, (key, value) in enumerate(info):
            text = f"{key}: {value}"
            cv2.putText(
                self._current_frame, text,
                (10, self._frame_height - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

    def _create_information_text(self, success):
        """ Создать информацию о трекинге для текущего кадра 
        Returns
        -------
        [(str, str), ...] - [(Название характеристики, значение)...]
        """
        return [
            ("Tracker", self._tracker),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(self._fps.fps())),
            ("Number", self._amount_frame),
        ]

    def _check_commands(self):
        """ Проверить нет ли команд 
                - окончания просмотра 
                - сохранения кадра 
                - выбора области
                - очищения области
        Returns
        -------
        EXIT_SUCCESS - если конец просмотра 
        CONTINUE_PROCESSING - если продолжение просмотра 
        """
        key = cv2.waitKey(30) & 0xFF
        # Esc - выход
        if key == 27:
            return EXIT_SUCCESS
        if cv2.getWindowProperty(DEFAULT_FRAME_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return EXIT_SUCCESS
        if key == ord("s"):
            self._save_frame()
        elif key == ord("a"):
            self._select_box()
        elif key == ord("d"):
                self._delete_box()            
        elif key == ord("w"):
            cv2.waitKey(-1)
        return CONTINUE_PROCESSING

    def _save_frame(self):
        """ Сохранить кадр """
        try:
            os.makedirs(DIRECTORY_SAVING)
        except OSError:
            pass
        now = datetime.datetime.now()
        now = str(now.strftime("%Y-%m-%d_%H-%M-%S_"))
        cv2.imwrite(
            DIRECTORY_SAVING + now +
            DEFAULT_IMAGE_NAME, self._current_frame
        )

    def _select_box(self):
        """ Выбрать область для трекинга """
        self._box = cv2.selectROI(
            DEFAULT_FRAME_WINDOW_NAME, self._current_frame,
            fromCenter=False, showCrosshair=True
        )
        if self._box[2:4] == (0, 0):
            self._box = None
            return        
        self._writer.writerow({
            "frame": self._amount_frame,
            "logs": "Select and update box"
        })
        self._tracker = OPENCV_OBJECT_TRACKERS[self._tracker_name]()
        self._tracker.init(self._current_frame, self._box)
        self._fps = FPS().start()

    def _delete_box(self):
        """ Удалить область трекинга """
        self._box = None    
        self._writer.writerow({
            "frame": self._amount_frame,
            "logs": "Clear box"
        })
