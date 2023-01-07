#the actual code which runs on dennis 
import os 
from pathlib import Path 
# import torch

# import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()
# setup_logger(name="mask2former")


# # import some common libraries
# import numpy as np
# import cv2
# # import some common detectron2 utilities
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog
# from detectron2.projects.deeplab import add_deeplab_config
# from pathlib import Path
# # import Mask2Former project
# from mask2former import add_maskformer2_config

# import pickle 
# import random 
# import pycocotools.mask as mask_util
# import numpy as np 
# from einops import rearrange, reduce, repeat

def read_txt(path):
    with open(path) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

txt_path = 'trainlist_brs1_atleast_32frames_smooth_strt_end_idx_info_158_pids_no_uav.txt'

lines = read_txt(txt_path)

data = []
for line in lines:
    l = line.split(' ')
    data.append([l[0], int(l[1]), int(l[2]), int(l[3])])

for d in data:
    v_dir, label, start, end = d[0],d[1],d[2],d[3]
    f_ids = sorted(os.listdir(v_dir))
    
    filtered_f_ids = []
    for f_id in f_ids:
        id = f_id.split('.')[0].split('_')[-1]
        id = int(id)
        print("id is",id)
        break