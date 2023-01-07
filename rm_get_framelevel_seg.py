import os 
import numpy as np
import cv2
import pickle 
import random 
from einops import rearrange, reduce, repeat
import json 
ann_path = '/home/rmodi/crcv_work/stream_lined_occlusions/models/MOC-Detector/data/ucf24/UCF101v2-GT.pkl'

with open(ann_path, 'rb') as fid:
    ann = pickle.load(fid, encoding='iso-8859-1')

vids = ann['train_videos'][0]+ ann['test_videos'][0]
print(len(vids))

gttubes = ann['gttubes']


d= {}

for done, v in enumerate(vids):
    print("done", done+1, len(vids))
    d[v] = {}
    ac_tubes = gttubes[v]
    for c in ac_tubes.keys():
        tubes = ac_tubes[c]
        for t in tubes:
            n_f, _ = t.shape 
            for i in range(n_f):
                f_id, x1,y1,x2,y2 = t[i]
                if f_id not in d[v].keys():
                    d[v][int(f_id)] = []
                d[v][int(f_id)].append([int(x1),int(y1),int(x2),int(y2)])
with open("seg.json", "w") as outfile:
    json.dump(d, outfile)