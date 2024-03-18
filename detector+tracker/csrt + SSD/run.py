import cv2
import numpy as np
from flask import Flask, jsonify, request

from view import DronesDetector_post

app = Flask(__name__)

DETECTOR = DronesDetector_post()


@app.route('/', methods=['GET', 'POST'])
def sample():
    print("start")
    image_data = request.get_data()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    pred = DETECTOR.update(image, 5)
    return jsonify({
        "bboxes": pred,

    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)  # , port=80
