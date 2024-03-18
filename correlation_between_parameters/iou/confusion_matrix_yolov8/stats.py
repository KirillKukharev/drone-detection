import os
from ultralytics import YOLO
from PIL import Image
from helper import _get_iou


def _get_txt_area(file_label, w, h):
    with open(file_label) as f:
        for line in f:
            box = [float(x) for x in line.split()]
        return box[1] * w, box[2] * h, box[3] * w, box[4] * h


def update_stats(stats, flag, area):
    if area <= 10:
        stats[f'0-10_{flag}'] = stats.get(f'0-10_{flag}', 0) + 1
    elif 10 < area <= 20:
        stats[f'10-20_{flag}'] = stats.get(f'10-20_{flag}', 0) + 1
    elif 20 < area <= 30:
        stats[f'20-30_{flag}'] = stats.get(f'20-30_{flag}', 0) + 1
    elif 30 < area <= 50:
        stats[f'30-50_{flag}'] = stats.get(f'30-50_{flag}', 0) + 1
    elif 50 < area <= 100:
        stats[f'50-100_{flag}'] = stats.get(f'50-100_{flag}', 0) + 1
    elif 100 < area <= 150:
        stats[f'100-150_{flag}'] = stats.get(f'100-150_{flag}', 0) + 1
    elif 150 < area <= 200:
        stats[f'150-200_{flag}'] = stats.get(f'150-200_{flag}', 0) + 1
    elif 200 < area <= 500:
        stats[f'200-500_{flag}'] = stats.get(f'200-500_{flag}', 0) + 1
    elif 500 < area:
        stats[f'500+_{flag}'] = stats.get(f'500+_{flag}', 0) + 1

    return stats


def get_stats():
    stats = {}
    paths_data_test = [
        '../test/images/part_5/ch02_20210930180049_3/'
    ]

    for path_data_test in paths_data_test:
        for img_file in [os.path.join(path_data_test, f) for f in os.listdir(path_data_test)
                         if os.path.isfile(os.path.join(path_data_test, f))]:

            img = Image.open(img_file)

            label_file = img_file.replace('images', 'labels').replace('jpg', 'txt')
            model = YOLO('../train/weights/best.pt')
            w = img.width
            h = img.height
            results = model.predict(img, conf=0.3, iou=0.5)
            stats['Total'] = stats.get('Total', 0) + 1
            bbox_true = _get_txt_area(label_file, w, h)
            area = bbox_true[2] * bbox_true[3]
            if len(results) == 0:
                stats = update_stats(stats, '_not_detect', area)

            fount_detect = False

            for result in results:
                for box in result.boxes.cpu():
                    for xywh in box.xywh.numpy().astype(int):
                        print(_get_iou(bbox_true, xywh))
                        print(bbox_true, xywh)
                        if _get_iou(bbox_true, xywh) > 0.3:
                            stats = update_stats(stats, '_true', area)
                            fount_detect = True
                            break

            if not fount_detect:
                stats = update_stats(stats, '_false_detect', area)

    return stats


if __name__ == '__main__':
    stats = get_stats()

    print("{:<20} {:<15}".format('Area', 'Count true'))
    print(stats)
    for v in stats:
        name, val = v.keys(), v.items()
        print("{:<20} {:<15}".format(name, val))
