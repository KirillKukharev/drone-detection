from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import yaml


with open('config.yml' , 'r') as f:
    config =yaml.safe_load(f)['yolov5_deepsort']['tracker']

#Visualization parameters

DISP_TRACKS = config['disp_tracks']
DISP_OBJ_TRACK_BOX = config['disp_obj_track_box']
OBJ_TRACK_COLOR = tuple(config['obj_tack_color'])
OBJ_TRACK_BOX_COLOR = tuple(config['obj_track_box_color'])

# # Deep Sort Parameters (check config.yml for parameter descriptions)
# MAX_AGE = config['max_age']   
# N_INIT =config['n_init']    
# NMS_MAX_OVERLAP = config['nms_max_overlap']       
# MAX_COSINE_DISTANCE = config['max_cosine_distance']    
# NN_BUDGET = config['nn_budget']            
# OVERRIDE_TRACK_CLASS = config['override_track_class'] 
# EMBEDDER = config['embedder']
# HALF = config['half'] 
# BGR = config['bgr']
# EMBEDDER_GPU = config['embedder_gpu'] 
# EMBEDDER_MODEL_NAME = config['embedder_model_name']    
# EMBEDDER_WTS = config['embedder_wts']           
# POLYGON = config['polygon']              
# TODAY = config['today']                  

# Deep Sort Parameters
MAX_AGE = 5                 # Maximum number of frames to keep a track alive without new detections. Default is 30

N_INIT =1                  # Minimum number of detections needed to start a new track. Default is 3

NMS_MAX_OVERLAP = 1.0       # Maximum overlap between bounding boxes allowed for non maximal supression(NMS).
                            #If two bounding boxes overlap by more than this value, the one with the lower confidence score is suppressed. Defaults to 1.0.

MAX_COSINE_DISTANCE = 0.3   # Maximum cosine distance allowed for matching detections to existing tracks. 
                            #If the cosine distance between the detection's feature vector and the track's feature vector is higher than this value, 
                            # the detection is not matched to the track. Defaults to 0.2

NN_BUDGET = None            # Maximum number of features to store in the Nearest Neighbor index. If set to None, the index will have an unlimited budget. 
                            #This parameter affects the memory usage of the tracker. Defaults to None.

OVERRIDE_TRACK_CLASS = None  #Optional override for the Track class used by the tracker. This can be used to subclass the Track class and add custom functionality. Defaults to None.
EMBEDDER = "mobilenet"       #The name of the feature extraction model to use. The options are "mobilenet" or "efficientnet". Defaults to "mobilenet".
HALF = True                  # Whether to use half-precision floating point format for feature extraction. This can reduce memory usage but may result in lower accuracy. Defaults to True
BGR = False                   #Whether to use BGR color format for images. If set to False, RGB format will be used. Defaults to True.
EMBEDDER_GPU = True          #Whether to use GPU for feature extraction. If set to False, CPU will be used. Defaults to True.
EMBEDDER_MODEL_NAME = None   #Optional model name for the feature extraction model. If not provided, the default model for the selected embedder will be used.
EMBEDDER_WTS = None          # Optional path to the weights file for the feature extraction model. If not provided, the default weights for the selected embedder will be used.
POLYGON = False              # Whether to use polygon instead of bounding boxes for tracking. Defaults to False.
TODAY = None                 # Optional argument to set the current date. This is used to calculate the age of each track in days. If not provided, the current date is used.

class DeepSortTracker(): 

    def __init__(self):
        
        self.algo_name ="DeepSORT"
        self.object_tracker = DeepSort(max_age=MAX_AGE,
                n_init=N_INIT,
                nms_max_overlap=NMS_MAX_OVERLAP,
                max_cosine_distance=MAX_COSINE_DISTANCE,
                nn_budget=NN_BUDGET,
                override_track_class=OVERRIDE_TRACK_CLASS,
                embedder=EMBEDDER,
                half=HALF,
                bgr=BGR,
                embedder_gpu=EMBEDDER_GPU,
                embedder_model_name=EMBEDDER_MODEL_NAME,
                embedder_wts=EMBEDDER_WTS,
                polygon=POLYGON,
                today=TODAY)
    
        
    def display_track(self , track_history , tracks_current , img):

        for track in tracks_current:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            
            # Retrieve the current track location(i.e - center of the bounding box) and bounding box
            location = track.to_tlbr()
            bbox = location[:4].astype(int)
            bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

            # Retrieve the previous center location, if available
            prev_centers = track_history.get(track_id ,[])
            prev_centers.append(bbox_center)
            track_history[track_id] = prev_centers
            
            # Draw the track line, if there is a previous center location
            if prev_centers is not None and DISP_TRACKS == True:
                points = np.array(prev_centers, np.int32)
                cv2.polylines(img, [points], False, (51 ,225, 255), 2)

            if DISP_OBJ_TRACK_BOX == True: 
                cv2.rectangle(img,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255),1)
                cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

            