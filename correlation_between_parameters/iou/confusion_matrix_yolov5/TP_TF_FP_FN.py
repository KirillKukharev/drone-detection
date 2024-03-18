import os
from PIL import Image
from helper import _get_iou
import matplotlib.pyplot as plt
import torch
import os
import cv2
import yaml
from datetime import datetime

from utils.metrics import bbox_iou
from yaml.loader import SafeLoader
from argparse import ArgumentParser

# changing colors according to the number of examples in cell
annotate_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

# Path
ROOT = os.getcwd()
imgsz = 640

# Load the model
model = torch.hub.load('ultralytics/yolov5' , 'custom' , path = '/Users/model.pt' , force_reload = True)
model.conf = 0.45
model.iou = 0.15

main_dir_imgs = "/Users/images/"
main_dir_labels = "/Users/labels/"
list_of_directories = ["ch02_20210930185126", "...", "ch02_20210930180049_3"]

def run():
    stats = {}
    print(f"started at {datetime.now()}")
    total_number_of_imgs = 0
    total_number_of_imgs_not = 0
    total_number_of_imgs_true = 0
    total_number_of_imgs_false = 0


    size_imgs_less_than_10_not = 0
    size_imgs_less_than_10_true = 0
    size_imgs_less_than_10_false = 0

    size_imgs_between_10_and_50_not = 0
    size_imgs_between_10_and_50_true = 0
    size_imgs_between_10_and_50_false = 0

    size_imgs_between_50_and_100_not = 0
    size_imgs_between_50_and_100_true = 0
    size_imgs_between_50_and_100_false = 0

    size_imgs_between_100_and_500_not = 0
    size_imgs_between_100_and_500_true = 0
    size_imgs_between_100_and_500_false = 0

    size_imgs_between_500_and_1000_not = 0
    size_imgs_between_500_and_1000_true = 0
    size_imgs_between_500_and_1000_false = 0

    size_imgs_over_than_1000_not = 0
    size_imgs_over_than_1000_true = 0
    size_imgs_over_than_1000_false = 0
    for dirs in list_of_directories:
        prime_dir_imgs = os.path.join(main_dir_imgs, dirs)
        prime_dir_labels = os.path.join(main_dir_labels, dirs)
        for img in os.listdir(prime_dir_imgs):
                name = img.split(".")[0]
                if img != ".DS_Store" and img.split(".")[1] =="jpg" :
                    # read image
                    img_d = cv2.imread(os.path.join(prime_dir_imgs, img))

                    # get parameters of image
                    img_d = cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB)

                    # get drones bbox values from image
                    square_of_drone = []
                    truth = torch.tensor([list(map(float, x.split())) for x in open(prime_dir_labels +"/" + name + ".txt").readlines()])
                    for k in range(len(truth)):
                        truth[k][3] = truth[k][3] * 1920
                        truth[k][4] = truth[k][4] * 1080
                        square_of_drone.append(int(truth[k][3] * truth[k][4]))

                        truth[k][1] = truth[k][1] * 1920
                        truth[k][2] = truth[k][2] * 1080

                    # make prediction
                    result = model(img_d, size=imgsz)
                    total_number_of_imgs+=len(truth)

                    # check the empty prediction (no drone on image)
                    if len(result.xywh[0]) == 0 and len(truth)>0:
                       for square in range(len(square_of_drone)):
                           total_number_of_imgs_not += 1
                           if square_of_drone[square] <= 10:
                               size_imgs_less_than_10_not+=1
                           elif 10 < square_of_drone[square] <=50:
                               size_imgs_between_10_and_50_not +=1
                           elif 50 < square_of_drone[square] <= 100:
                               size_imgs_between_50_and_100_not+=1
                           elif 100 < square_of_drone[square] <= 500:
                               size_imgs_between_100_and_500_not+=1
                           elif 500 < square_of_drone[square] <= 1000:
                               size_imgs_between_500_and_1000_not+=1
                           else:
                               size_imgs_over_than_1000_not+=1
                    else:
                        # if there are drone on the image, check each predicted bbox
                        for r in range(len(result.xywh[0])):
                            # find max val of iou for each prediction 
                            iou = float(max(bbox_iou(result.xywh[0][r][None,:4],truth[:, 1:])))

                            # get size from prediction
                            squar = result.xywh[0][r][2] * result.xywh[0][r][3]

                            # if iou more than 0.5, increase value of the target square of drone
                            if iou > 0.1:
                                total_number_of_imgs_true+=1
                                if squar <= 10:
                                        size_imgs_less_than_10_true += 1
                                elif 10 < squar <= 50:
                                        size_imgs_between_10_and_50_true += 1
                                elif 50 < squar <= 100:
                                        size_imgs_between_50_and_100_true += 1
                                elif 100 < squar <= 500:
                                        size_imgs_between_100_and_500_true += 1
                                elif 500 < squar <= 1000:
                                        size_imgs_between_500_and_1000_true += 1
                                else:
                                        size_imgs_over_than_1000_true += 1
                            else:
                                # if iou too small, then make resulted prediction and increase value of drones for target square
                                total_number_of_imgs_false+=1
                                if squar <= 10:
                                    size_imgs_less_than_10_false += 1
                                elif 10 < squar <= 50:
                                    size_imgs_between_10_and_50_false += 1
                                elif 50 < squar <= 100:
                                    size_imgs_between_50_and_100_false += 1
                                elif 100 < squar <= 500:
                                    size_imgs_between_100_and_500_false += 1
                                elif 500 < squar <= 1000:
                                    size_imgs_between_500_and_1000_false += 1
                                else:
                                    size_imgs_over_than_1000_false += 1


        # plot confusion matrix
        confusion_matrix = [[total_number_of_imgs_true, total_number_of_imgs_false, total_number_of_imgs_not],
                            [size_imgs_less_than_10_true, size_imgs_less_than_10_false, size_imgs_less_than_10_not],
                            [size_imgs_between_10_and_50_true, size_imgs_between_10_and_50_false, size_imgs_between_10_and_50_not],
                            [size_imgs_between_50_and_100_true, size_imgs_between_50_and_100_false, size_imgs_between_50_and_100_not],
                            [size_imgs_between_100_and_500_true, size_imgs_between_100_and_500_false, size_imgs_between_100_and_500_not],
                            [size_imgs_between_500_and_1000_true, size_imgs_between_500_and_1000_false, size_imgs_between_500_and_1000_not],
                            [size_imgs_over_than_1000_true, size_imgs_over_than_1000_false, size_imgs_over_than_1000_not]]
        classes_y = ["Total", "size<=10", "size (10, 50]", "size (50, 100]", "size (100, 500]", "size (500, 1000]", "size >1000"]
        classes_x = ["True", "False", "Not"]
        fig, ax = plt.subplots()
        im = ax.imshow(confusion_matrix, cmap="Blues", vmin=0, vmax=20)
        ax.set_xticks(range(len(classes_x)))
        ax.set_yticks(range(len(classes_y)))

        ax.set_xticklabels(classes_x)  
        ax.set_yticklabels(classes_y) 
        ax.set_xlabel("Предсказание") 
        ax.set_ylabel("Размер изображений")  
        for i in range(len(classes_y)):  
            for j in range(len(classes_x)): 
                value = confusion_matrix[i][j]
                ax.text(j, i, value, ha="center", va="center", color="w" if value > 10 else "k")

        # display matrix
        # plt.show()

        # save matrix in confusion_matrix_conf.png
        plt.savefig("confusion_matrix_conf.png")
        print(f"finished {dirs} at {datetime.now()}")

run()