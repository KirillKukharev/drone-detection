import cv2

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

DIRECTORY_SAVING = "/Volumes/MYUSB/видео/ch01_20210930185055_3"
DEFAULT_IMAGE_NAME = "Tracker_frame_video.png"

DEFAULT_FRAME_WINDOW_NAME = "Frame"
DEFAULT_CSV_NAME = "Boxes.csv"

EXIT_SUCCESS = 0
# EXIT_FAILURE = 1
CONTINUE_PROCESSING = 2