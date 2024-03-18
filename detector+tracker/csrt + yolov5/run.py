import cv2
import numpy as np
from flask import Flask, jsonify, request

from view import DronesDetector_post
from datetime import datetime, date, time
#import io
import os
import time

os.environ['CUDA_LAUNCH_BLOCKING']="1"

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print("Memory usage:")
    print("Allocated: ", round(torch.cuda.memory_allocated(0)/1024**3,1), "GB")
    print("Cached:  ", round(torch.cuda.memory_reserved(0)/1024**3,1), "GB")

app = Flask(__name__)

DETECTOR = DronesDetector_post()

@app.route('/', methods=['GET', 'POST'])
def sample():
    file = request.files['image']
    date_time_get = datetime.utcnow()
    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    data = request.form.to_dict()
    pred = DETECTOR.update(frame, int(data['number']))
    date_time_processed = datetime.utcnow()
    time_delta_1 = (date_time_processed-date_time_get).total_seconds()
    print(f"frame_num={int(data['number'])},total_time = {time_delta_1}")
    return jsonify({
        "bboxes": pred,

    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)  # , port=80

