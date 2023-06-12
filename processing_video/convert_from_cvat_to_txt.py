# python3 convert_from_cvat_to_txt.py --input labels.xml

import csv
import datetime
import os
import xml.etree.ElementTree as xml
import argparse

DIRECTORY_SAVING = "./"

class Converter:
    """ Конвертация разметки.

    Parameters
    ----------
    path_in_file : str
        Путь к файлу с разметкой.

    Attributes
    ----------
    _path_in_file : str
        Путь к файлу с разметкой.

    Method
    -------
    convert_xml_to_my_txt()
        Конвертация xml файла с разметкой CVAT for video 1.1 в csv.
    """

    def __init__(self, path_in_file):
        self._path_in_file = path_in_file

    def convert_xml_to_my_txt(self):
        """
        Конвертация xml файла с разметкой CVAT for video 1.1 (https://cvat.org)
        в csv (frame, x, y, w, h, logs - номер кадра, координаты 
        левого верхнего угла, ширина, высота, логи).
        """
        now = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_"))
        name_input_file = os.path.basename(self._path_in_file).split(".")[0]
        txt_file_name = DIRECTORY_SAVING + now + name_input_file + ".csv"
        # fieldnames = ["frame", "x", "y", "w", "h", "logs"]
        fieldnames = ['frame', 'x', 'y', 'w', 'h']
        with open(txt_file_name, "w", newline="") as txt_file:
            writer = csv.DictWriter(txt_file, fieldnames=fieldnames)
            writer.writeheader()
            xml_tree = xml.parse(self._path_in_file)
            xml_root = xml_tree.getroot()
            for box in xml_root.findall(".//box[@outside='0']"):
                frame = int(box.get("frame"))
                x = int(float(box.get("xtl")))
                y = int(float(box.get("ytl")))
                w = int(float(box.get("xbr"))) - x
                h = int(float(box.get("ybr"))) - y

                writer.writerow(
                {"frame": frame, "x": x, "y": y, "w": w, "h": h}
            )
        self._sort_txt(txt_file_name, fieldnames)

    def _sort_txt(self, txt_file_name, fieldnames):
        """Сортировка разметки csv по колонке frame

        Parameters
        ----------
        txt_file_name : str
            Путь к txt файлу;
        fieldnames : list
            Список имен колонок:
                - str Имя колонки.            
        """
        sorted_list = []
        with open(txt_file_name, newline="") as txt_file:
            reader = csv.DictReader(txt_file)
            sorted_list = sorted(
                reader,
                key=lambda row: (int(row["frame"])),
                reverse=False
            )
        with open(txt_file_name, "w", newline="") as txt_file:
            writer = csv.DictWriter(txt_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_list)

        with open(txt_file_name, 'r') as f_in, open('annotation.txt', 'w') as f_out:
            # 2. Read the CSV file and store in variable
            content = f_in.read().replace(',', ' ')
            # 3. Write the content into the TXT file
            f_out.write(content)

def parse():
    parser = argparse.ArgumentParser(
        description="This example demonstrates converting."
    )
    parser.add_argument(
        "-input", "--input", type=str, help="path to input file"
    )
    return parser.parse_args()


args = vars(parse())
if not args.get("input", False):
    print("Нужно указать путь к исходному файлу. Параметр -input или --input.")
else:
    converter = Converter(args["input"])
    convert = converter.convert_xml_to_my_txt()