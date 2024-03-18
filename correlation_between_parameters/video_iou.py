import cv2
import pandas as pd

# read annotation path
data = pd.read_csv("/Users/ch02_20210930185126_1.csv")

# read video path
video = cv2.VideoCapture("/Users/kirill/Desktop/обученные модели/ch02_20210930185126_1_v5n.mp4")

# process each frame of the video
while video.isOpened():
    ret, frame = video.read()
    # if frame is ok, then continue processing
    if ret:
        # get the number of the current frame
        frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        frame_data = data[data["frame"] == frame_number]
        # iterate for each object on the frame
        for index, row in frame_data.iterrows():
            x = int(row["x"])
            y = int(row["y"])
            w = int(row["w"])
            h = int(row["h"])
            # Plot rectangle on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(index), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # display frame
        cv2.imshow("Video", frame)
        # continue process video and go to the next frame
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    else:
        break

# release resources
video.release()
cv2.destroyAllWindows()