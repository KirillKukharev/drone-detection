# This file checks that the labelling corresponds to the given image

import cv2

image = cv2.imread('path/to/image.jpg')
with open('path/to/annotation.txt', 'r') as f:
    for line in f.readlines():
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        x_min = int((x_center - width / 2) * image.shape[1])
        y_min = int((y_center - height / 2) * image.shape[0])
        x_max = int((x_center + width / 2) * image.shape[1])
        y_max = int((y_center + height / 2) * image.shape[0])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)