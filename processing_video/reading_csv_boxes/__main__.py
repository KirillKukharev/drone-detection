import argparse

from reader import Reader

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, help="path to input video file")
    parser.add_argument("-f", "--csv", type=str, help="path to input csv file")
    parser.add_argument("-sv", "--sv", type=bool, help="saving video")
    return parser.parse_args()


args = vars(parse())
if not args.get("video", False):
    print("Нужно указать путь к видео. Параметр -v или --video.")
if not args.get("sv", False):
    print("Err")
if not args.get("csv", False):
    print("Нужно указать путь к csv. Параметр -f или --csv.")
else:
    reader = Reader(args["video"], args["csv"], args["sv"])
    reader.run()