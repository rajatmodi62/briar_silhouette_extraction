#ccode to extract the h5 file 
# apparently that is only the thing which is remaining 

import os 
from pathlib import Path 
import torch
import h5py
#the actual code which runs on dennis 
import os 
from pathlib import Path 
import torch

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")
import copy

# import some common libraries
import numpy as np
import cv2
# import some common detectron2 utilities
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

# p =  '/data/project/clip_extraction_brs_bts_v2/afs_brs1_static/brs_bts_version_1/BRS1/G00021/field/500m/wb/G00021_set1_struct_1623185822092_45efd745/frame_000000_000039_clip_0000/clip.h5'

# clip_data = h5py.File(p, 'r')
# data = clip_data['data']

test_dir = '/data/project/clip_extraction_brs_bts_v2/afs_brs1_static/brs_bts_version_1/BRS1/'
filtered_f_ids = []
for p in os.walk(str(test_dir)):
    v_folder = p[0]
    # v_folder = v_folder.split('/')[2:]
    # v_folder = '/'.join(v_folder)

    f_ids = p[2]
    
    for f_id in f_ids:
        if '.h5' in f_id:
            filtered_f_ids.append([v_folder,f_id])
            # print("true")
print("len of", len(filtered_f_ids),v_folder)
print(filtered_f_ids[0])


cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
predictor = DefaultPredictor(cfg)

root = Path('/data/project/rm_silhouette')
score_thresh = 0.5
filtered_f_ids = sorted(filtered_f_ids)
n = len(filtered_f_ids)

for done, item in enumerate(filtered_f_ids[n//2:]):
    try:
        # print("done", done, "/", len(filtered_f_ids))
        print("trying")
        # print("v_folder", v_folder)
        v_folder, h5_path = item
        src_path = Path(v_folder)/h5_path
        clip_data = h5py.File(str(src_path), 'r')
        data = clip_data['data']

        n_frames = data.shape[0]
        for f_id in range(n_frames):
            print("done", done, "/", len(filtered_f_ids), f_id,"/", n_frames)
            id = copy.deepcopy(f_id)
            f_id = str(f_id)+'.jpg'
            save_id = h5_path.split('.')[0]

            save_v_folder = v_folder.split('/')[3:]
            save_v_folder = '/'.join(save_v_folder)
        # print("save v ", save_v_folder)
            (root/save_v_folder/save_id).mkdir(exist_ok= True, parents =True)
            save_path = root/save_v_folder/save_id/f_id
    
            # im = cv2.imread(str(src_path))
            # print("read im", im.shape)
            im = data[id]
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
            cv2.imwrite(str(save_path), overall_mask*255)
            # exit(1)
    except:
        print("some error")