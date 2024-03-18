import csv
import datetime
import os

import cv2

from constants import *


class Reader:
    """ Отбражение рамок на видео согласно аннотации

    Parameters
    ----------
    path_video: str
        Путь к файлу с видео,
    path_csv: str
        Путь к файлу с аннотацией,
    saving_video: boolean
        Сохранять ли видео с рамками (default=False)

    Attributes
    ----------
    _path_csv: str
        Путь к файлу с аннотацией,
    _saving_video: boolean
        Сохранять ли видео с рамками,
    _capture: VideoCapture
        Видео,
    _current_frame: array([...], dtype=uint8)
        Текущий кадр,
    _amount_frame: int
        Номер кадра,
    _frame_height: int
        Высота кадра,
    _frame_width: int
        Ширина кадра,
    _out_video: VideoWriter
        Видео с контурами
    """

    def __init__(self, path_video, path_csv, saving_video=False):
        self._path_csv = path_csv
        self._saving_video = saving_video
        self._capture = cv2.VideoCapture(path_video)
        _, self._current_frame = self._capture.read()
        self._amount_frame = 0
        self._frame_height, self._frame_width = self._current_frame.shape[:2]
        self._out_video = None
        if self._saving_video:
            self._init_out_video()

    def _init_out_video(self):
        """ Инициализировать атрибуты с видео результатом """
        try:
            os.makedirs(DIRECTORY_SAVING)
        except OSError:
            pass
        now = datetime.datetime.now()
        now = str(now.strftime("%Y-%m-%d_%H-%M-%S_"))
        framerate = 25
        self._out_video = cv2.VideoWriter(
            DIRECTORY_SAVING + now + DEFAULT_VIDEO_NAME,
            cv2.VideoWriter_fourcc(*"mp4v"),
            framerate, (self._frame_width, self._frame_height)
        )

    def run(self):
        """ Запустить отрисовку рамок на видео """
        with open(self._path_csv, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            row = next(reader, None)
            while(True):
                if self._current_frame is None:
                    break
                # if not row is None and int(row["frame"]) == self._amount_frame:
                #     row = next(reader, None)
                #     print("hz1")
                if not row is None and int(row["frame"]) < self._amount_frame:
                    row = next(reader, None)
                    print("hz2"+str(row['frame']))
                if not row is None and int(row["frame"]) == self._amount_frame and row['logs']=="":
                        cv2.rectangle(
                            self._current_frame, (int(float(row["x"])), int(float(row["y"]))),
                            (int(float(row["x"])) + int(float(row["w"])),
                             int(float(row["y"])) + int(float(row["h"]))),
                            (0, 255, 255),
                            1
                        )
                        row = next(reader, None)
                        continue
                self._drow_count()
                cv2.namedWindow(DEFAULT_FRAME_WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(DEFAULT_FRAME_WINDOW_NAME, self._current_frame)
                self._save_video()
                if self._check_commands() == EXIT_SUCCESS:
                    break
                _, self._current_frame = self._capture.read()
                self._amount_frame += 1

    def _drow_count(self):
        """ Нарисовать счетчик """
        coordinate_x_place_text = int(self._frame_width / 2) - 150
        COORDINATE_Y_PLACE_TEXT = 100
        font_scale = 1
        text_color = (0, 255, 255)
        cv2.putText(
            self._current_frame, "Number frame: " + str(self._amount_frame),
            (coordinate_x_place_text, COORDINATE_Y_PLACE_TEXT),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness=2
        )

    def _check_commands(self):
        """ Проверить нет ли команд
                - окончания просмотра
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
        return CONTINUE_PROCESSING

    def _save_video(self):
        """ Сохранить результат """
        if self._saving_video:
            self._out_video.write(self._current_frame)