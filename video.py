import os
import cv2

path = "work_dirs/VILane/vis/JPEGImages/5_Road001_Trim008_frames/"
f = os.listdir(path)
f = sorted(f)
print(f)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vout = cv2.VideoWriter('demo.mp4', fourcc, 5.0, (1920, 1080))
for i in f:
    # import pdb; pdb.set_trace()
    vis = cv2.imread(path+i)
    vout.write(vis)

vout.release()

