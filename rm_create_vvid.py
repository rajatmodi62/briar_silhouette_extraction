import os 
import numpy as np 
import cv2 
from pathlib import Path
p = Path('./v2')
v_name = 'v2.avi'

paths = sorted(os.listdir(str(p)))
frames = []

for done, f_name in enumerate(paths):
    print("done",done,len(paths))

    read_path = p/f_name 
    img = cv2.imread(str(read_path))
    frames.append(img)
h,w,_ = frames[0].shape

out = cv2.VideoWriter(v_name,cv2.VideoWriter_fourcc(*'DIVX'), 15, (w,h))
for done, f in enumerate(frames):
    print("done", done, len(frames))
    out.write(f)
out.release()
