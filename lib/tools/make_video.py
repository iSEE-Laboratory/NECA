import os
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True, type=str)
parser.add_argument("--size", default=512, type=int, help="image size")

args = parser.parse_args()


size = (args.size, args.size)  # 图片的尺寸
path = args.path

if os.path.exists(os.path.join(path,'result.mp4')):
    os.remove(os.path.join(path,'result.mp4'))

files = os.listdir(path)

files.sort(key=lambda x:x) #int(x[:4])

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
videowrite = cv2.VideoWriter(os.path.join(path, 'result.mp4'), fourcc, 25, size)


for filename in files:
    filename = os.path.join(path, filename)
    img = cv2.imread(filename)
    if img is None:
        print(filename + " is error!")
        continue
    videowrite.write(img)
videowrite.release()
print('end!')