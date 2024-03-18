# cd to yolov5-deepsort
# python3.8 main.py

# с записью видео
import cv2
import time
import os
import sys
import yaml

from src.detector import YOLOv5Detector
from src.tracker import DeepSortTracker
from src.dataloader import cap

# Parameters from config.yml file
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)['yolov5_deepsort']['main']

# Add the src directory to the module search path
sys.path.append(os.path.abspath('src'))

# Get YOLO Model Parameter
YOLO_MODEL_NAME = config['model_name']

# Visualization Parameters
DISP_FPS = config['disp_fps']
DISP_OBJ_COUNT = config['disp_obj_count']

object_detector = YOLOv5Detector(model_name=YOLO_MODEL_NAME)
tracker = DeepSortTracker()

track_history = {}  # Define a empty dictionary to store the previous center locations for each track ID

# Create a video writer object
output_file = 'output.avi'  # Specify the output file name
codec = cv2.VideoWriter_fourcc(*'XVID')  # Specify the codec
fps = 30  # Specify the frame rate
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # Specify the frame size
video_writer = cv2.VideoWriter(output_file, codec, fps, frame_size)  # Create a video writer object

while cap.isOpened():

    success, img = cap.read()  # Read the image frame from data source

    start_time = time.perf_counter()  # Start Timer - needed to calculate FPS

    # Object Detection
    results = object_detector.run_yolo(img)  # run the yolo v5 object detector
    detections, num_objects = object_detector.extract_detections(results, img, height=img.shape[0], width=img.shape[
        1])  # Plot the bounding boxes and extract detections (needed for DeepSORT) and number of relavent objects detected

    # Object Tracking
    tracks_current = tracker.object_tracker.update_tracks(detections, frame=img)  #
    tracker.display_track(track_history, tracks_current, img)

    # FPS Calculation
    end_time = time.perf_counter()
    total_time = end_time - start_time
    fps = 1 / total_time

    # Descriptions on the output visualization
    cv2.putText(img, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, f'MODEL: {YOLO_MODEL_NAME}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, f'TRACKED CLASS: {object_detector.tracked_class}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1)
    cv2.putText(img, f'TRACKER: {tracker.algo_name}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, f'DETECTED OBJECTS: {num_objects}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('img', img)

    # Write the frame to the output file
    video_writer.write(img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release and destroy all windows before termination
cap.release()
video_writer.release()  # Release the video writer object

cv2.destroyAllWindows()