import cv2
import numpy as np
import torch
import yaml

with open('config.yml' , 'r') as f:
    config =yaml.safe_load(f)['yolov5_deepsort']['detector']

TRACKED_CLASS = config['tracked_class']
DOWNSCALE_FACTOR = config['downscale_factor']
CONFIDENCE_THRESHOLD = config['confidence_threshold']
DISP_OBJ_DETECT_BOX = config['disp_obj_detect_box']

class YOLOv5Detector(): 

    def __init__(self, model_name):

        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: " , self.device)

        self.downscale_factor = DOWNSCALE_FACTOR  # Reduce the resolution of the input frame by this factor to speed up object detection process
        self.confidence_threshold = CONFIDENCE_THRESHOLD # Minimum theshold for the detection bounding box to be displayed
        self.tracked_class = TRACKED_CLASS

    def load_model(self , model_name):  # Load a specific yolo v5 model or the default model

        if model_name: 
            model = torch.hub.load('ultralytics/yolov5' , 'custom' , path = model_name , force_reload = True)
        else: 
            model = torch.hub.load('ultralytics/yolov5' , 'yolov5s' , pretrained = True)
        return model
 
    def run_yolo(self , frame): 
        self.model.to(self.device) # Transfer a model and its associated tensors to CPU or GPU
        frame_width = int(frame.shape[1]/self.downscale_factor)
        frame_height = int(frame.shape[0]/self.downscale_factor)
        frame_resized = cv2.resize(frame , (frame_width,frame_height))

        yolo_result = self.model(frame_resized)

        labels , bb_cord = yolo_result.xyxyn[0][:,-1] , yolo_result.xyxyn[0][:,:-1]
        
        return labels , bb_cord
        

    def class_to_label(self, x):

        return self.classes[int(x)]
        
    def extract_detections(self, results, frame, height, width):

        labels, bb_cordinates = results  # Extract labels and bounding box coordinates
        detections = []         # Empty list to store the detections later 
        class_count = 0         # Initialize class count for the frame 
        num_objects = len(labels)   #extract the number of objects detected
        x_shape, y_shape = width, height

        for object_index in range(num_objects):
            row = bb_cordinates[object_index]

            if row[4] >= self.confidence_threshold:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                
                if self.class_to_label(labels[object_index]) == self.tracked_class :
                    
                    if DISP_OBJ_DETECT_BOX: 
                        self.plot_boxes(x1 , y1 , x2 , y2 , frame)
                    x_center = x1 + ((x2-x1)/2)
                    y_center = y1 + ((y2 - y1) / 2)
                    conf_val = float(row[4].item())
                    feature = self.tracked_class

                    class_count+=1
                    
                    detections.append(([x1, y1, int(x2-x1), int(y2-y1)], row[4].item(), feature))
                    # We structure the detections in this way because we want the bbs expected to be a list of detections in the tracker, each in tuples of ( [left,top,w,h], confidence, detection_class) - Check deep-sort-realtime 1.3.2 documentation

        return detections , class_count
    
    def plot_boxes(self , x1 , y1 , x2 , y2 , frame):  
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
