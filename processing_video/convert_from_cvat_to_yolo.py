# source venv/bin/activate
# python3.8 frames_from_video

# Convert cvat annotation to yolo

import time
import os
import csv
import cv2

IMAGE_SAVING_DIR = "/Users/saving_images"
if not os.path.exists(IMAGE_SAVING_DIR):
    os.makedirs(IMAGE_SAVING_DIR)

start_time = time.time()

try:

    one_annotation_path = "/Users/annotations.csv"

    frame_count = 0
    with open(one_annotation_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        annotation_row = next(reader, None)
        while True:
            if annotation_row is None:
                break

            annotation_frame = int(annotation_row["frame"])
            if annotation_frame < frame_count:
                annotation_row = next(reader, None)
                continue
            print("Current frame " + str(frame_count))

    with open(one_annotation_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        with open(f'output_{i}.txt', 'w') as f:
            f.write(line)

    # read annotation
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

                # write image

                xtl = int(annotation_row["x"])
                ytl = int(annotation_row["y"])
                w = int(annotation_row["w"])
                h = int(annotation_row["h"])
                xbr = xtl + w
                ybr = ytl + h

                # check annotation
                if xtl < 0:
                    xtl = 0
                if ytl < 0:
                    ytl = 0
                if frame_width <= xbr:
                    xbr = frame_width - 1
                if frame_height <= ybr:
                    ybr = frame_height - 1

                file_name = f"ch01_20200605115731-frame {frame_count:08d}-num xtl({xtl})_ytl({ytl})_w({w})_h({h}).jpg"

                cv2.imwrite(
                    os.path.join(IMAGE_SAVING_DIR, file_name),
                    current_frame
                )
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