#the actual code which runs on dennis 
import os 



# print(len(dirs))
# exit(1)
from pathlib import Path 
import torch

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")
print("fetching dirs")
dirs = []
# paths = os.listdir('/data/project/bts_evaluation_dataset/gallery')
# for p in paths:
#     dirs.append(str(Path('/data/project/bts_evaluation_dataset/gallery')/p))
print("before",len(dirs))
paths= os.listdir('/data/project/bts_evaluation_dataset/probe')
for p in paths:
    dirs.append(str(Path('/data/project/bts_evaluation_dataset/probe')/p))
print("after",len(dirs))
dirs = sorted(dirs)
# # import some common libraries
import numpy as np
import cv2
# # import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from pathlib import Path
# import Mask2Former project
from mask2former import add_maskformer2_config

import pickle 
import random 
import pycocotools.mask as mask_util
import numpy as np 
from einops import rearrange, reduce, repeat

# def read_txt(path):
#     with open(path) as f:
#         lines = f.readlines()
#         lines = [line.strip() for line in lines]
#     return lines

# txt_path = 'trainlist_brs2_atleast_32_frames_smooth_strt_end_idx_info_195_pids_no_uav.txt'

save_root = Path('/data/project/rm_silhouette/test_set')

# root = Path('/data/project/rm_silhouette')
# n_frames = 100 # to_sample
score_thresh = 0.5
alpha = 0.5

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
predictor = DefaultPredictor(cfg)


# lines = read_txt(txt_path)
data = dirs
# data = []
# for done, line in enumerate(lines):
#     try:
#         print("done", done, "/", len(lines))
#         l = line.split(' ')
#         data.append([l[0], int(l[1]), int(l[2]), int(l[3])])
#     except:
#         print("some error")
for done, img_path in enumerate(data):
    try:
            print("done",done, "/", len(data))
            # print("img_path",img_path)
            folder_name = img_path.split('/')[-2]
            (save_root/folder_name).mkdir(exist_ok = True, parents = True)
            img_name = img_path.split('/')[-1]
            # img_path = Path(v_dir)/f_id
            im = cv2.imread(str(img_path))
            h,w, _ = im.shape
            print("read im", im.shape)
            if max(h,w)> 500:
                print("continuing")
                continue
            
            # exit(1)
            outputs = predictor(im)

            # v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
            
            # instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
            
            rm_outputs = outputs['instances'].to("cpu")
            
            boxes = rm_outputs.pred_boxes
            scores = rm_outputs.scores
            classes = rm_outputs.pred_classes
            masks =rm_outputs.pred_masks
            masks = rearrange(masks, 'n h w -> h w n')

            # rle = mask_util.encode(np.array(masks, order="F", dtype="uint8"))
            
            c = {'boxes':boxes,\
                'scores':scores,\
                'classes':classes,\
                
            }
            # print("hello",scores.shape, classes.shape, masks.shape,classes)
            
            n_preds = classes.shape[0]

            # h,w,n = masks.shape 
            # overall_mask = (np.zeros_like((h,w,1))).astype(np.float32)
            overall_mask = None
            for i in range(n_preds):
                c = classes[i] 
                if c==0:
                    mask = masks[:,:,i].numpy()
                    mask = repeat(mask, 'h w -> h w c', c = 1)
                    if scores[i] > score_thresh:
                        # print(mask.shape, overall_mask.shape)
                        if overall_mask is None:
                            overall_mask = mask
                        else:
                            overall_mask+=mask#.astype(np.float)
            overall_mask = (overall_mask > 0)*1
            # print("overall mask",overall_mask.shape)
            save_path = save_root/folder_name/img_name
            cv2.imwrite(str(save_path), overall_mask*255)
            print("saved")
            
            with open('probe.txt', 'w') as f:
                f.write(str(done))
            f.close()
            # exit(1)
    except:
        print("some exception occcured")
