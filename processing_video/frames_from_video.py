# This file extracts frames from the video according to a preset layout
# python3.8 frames_from_video


import time
import os
import csv
import cv2

IMAGE_SAVING_DIR = "path/to/saving_images"
LABEL_SAVING_DIR = "path/to/saving_labels"
if not os.path.exists(IMAGE_SAVING_DIR):
    os.makedirs(IMAGE_SAVING_DIR)

start_time = time.time()

try:

    video_path = f"path/to/video.mp4"
    annotation_path = "path/to/annotations.csv"

    capture = cv2.VideoCapture(video_path)
    _, current_frame = capture.read()
    frame_height, frame_width = current_frame.shape[:2]

    frame_count = 0

    # Чтение разметки
    with open(annotation_path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            annotation_row = next(reader, None)
            count_object = 0
            while True:
                if current_frame is None:
                    break
                if annotation_row is None:
                    break

                annotation_frame = int(annotation_row["frame"])
                if annotation_frame < frame_count:
                    annotation_row = next(reader, None)
                    continue

                print("Current frame " + str(frame_count))

                if frame_count < annotation_frame:
                    _, current_frame = capture.read()
                    frame_count += 1
                    count_object = 0
                    continue

                # Запись изображения

                xtl = int(annotation_row["x"])
                ytl = int(annotation_row["y"])
                w = int(annotation_row["w"])
                h = int(annotation_row["h"])
                xbr = xtl + w
                ybr = ytl + h

                new_xc = (xtl + w / 2) / 1920
                new_yc = (ytl + h / 2) / 1080
                new_w = w / 1920
                new_h = h / 1080

                # Проверка аннотации
                if xtl < 0:
                    xtl = 0
                if ytl < 0:
                    ytl = 0
                if frame_width <= xbr:
                    xbr = frame_width - 1
                if frame_height <= ybr:
                    ybr = frame_height - 1


                file_name_img = f"{frame_count:08d}.jpg"
                file_name_label = f"{frame_count:08d}.txt"

                cv2.imwrite(
                    os.path.join(IMAGE_SAVING_DIR, file_name_img),
                    current_frame
                )
                with open(f"{LABEL_SAVING_DIR}/{file_name_label}", 'w') as file:
                    file.write(f"{0} {new_xc} {new_yc} {new_w} {new_h}")


                count_object += 1
                print("writed frame " + str(frame_count))

                annotation_row = next(reader, None)

except KeyboardInterrupt:
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    raise

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))