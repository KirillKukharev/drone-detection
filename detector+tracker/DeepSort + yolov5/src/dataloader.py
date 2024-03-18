import cv2
import yaml

with open('config.yml' , 'r') as f:
    config =yaml.safe_load(f)['yolov5_deepsort']['dataloader']

# Data Source Parameters
DATA_SOURCE = config['data_source']   
WEBCAM_ID = config['webcam_id']  
DATA_PATH = config['data_path']  
FRAME_WIDTH = config['frame_width']
FRAME_HEIGHT = config['frame_height'] 

# Select Data Source 
if DATA_SOURCE == "webcam": 
    cap = cv2.VideoCapture(WEBCAM_ID)
elif DATA_SOURCE == "video file": 
    cap = cv2.VideoCapture(DATA_PATH)
else: print("Enter correct data source")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

