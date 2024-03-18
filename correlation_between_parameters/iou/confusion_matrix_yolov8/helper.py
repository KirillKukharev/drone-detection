def get_max_index(array: list) -> int:
    max_value = max(array)
    return array.index(max_value)



def _yolobbox2bbox(box: tuple) -> tuple:
    """
    Перевод box из (x, y, w, h) в ( xmin, ymin, xmax, ymax )

    :param box: массив  с координатами (x, y, w, h)
    :return: xmin, ymin, xmax, ymax
    """
    x, y, w, h = box
    xmin, ymin = x - w / 2, y - h / 2
    xmax, ymax = x + w / 2, y + h / 2
    return xmin, ymin, xmax, ymax


def _get_iou(box1, box2):
    """
    Данный метод возвращает IoU между box1 и box2.

    :param box1: массив numpy с координатами (x, y, w, h)
    :param box2: массив numpy с координатами (x, y, w, h)
    :return: iou
    """

    x11, y11, x21, y21 = _yolobbox2bbox(box1)
    x12, y12, x22, y22 = _yolobbox2bbox(box2)

    yi1 = max(y11, y12)
    xi1 = max(x11, x12)
    yi2 = min(y21, y22)
    xi2 = min(x21, x22)
    inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)

    # Рассчитываем площадь объединения по формуле: Union(A,B) = A + B - Inter(A,B)
    box1_area = (x21 - x11) * (y21 - y11)
    box2_area = (x22 - x12) * (y22 - y12)
    union_area = box1_area + box2_area - inter_area
    # IoU
    iou = inter_area / union_area
    return iou


def get_most_similar_bbox(bboxes_detect, bbox_track):
    """
    :param bboxes_detect: bboxes определнные детектором
    :param bbox_track: bbox определнный трекером
    :return: most similar bbox и его iou
    """
    # Будем хранить информацию о самой схожей рамке по track кадрам
    most_similar_bbox = {'bbox': [], "distance": 1000}

    for bbox in bboxes_detect:
        if bbox_track:
            x, y, _, _ = bbox
            x_, y_, _, _ = bbox_track
            # Если track не пустой, то считаем растояние
            iou_bbox = ((x-x_)**2 + (y-y_)**2)**(1/2)

            if iou_bbox < most_similar_bbox.get('distance'):
                most_similar_bbox = {'bbox': bbox, "distance": iou_bbox}

        return most_similar_bbox