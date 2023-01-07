#get segmentation mask 
# Some basic setup:
# Setup detectron2 logger
import os 
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")

# import some common libraries
import numpy as np
import cv2
import torch
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

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
predictor = DefaultPredictor(cfg)

frame_dir = Path('/home/rmodi/manural_annotations/Mask2Former/G00036')

frame_paths = sorted(os.listdir(str(frame_dir)))


save_dir = Path('/home/rmodi/manural_annotations/Mask2Former/v2')
score_thresh = 0.5
alpha = 0.5
for done, p in enumerate(frame_paths):
    print("done",done+1,"/", len(frame_paths))

    img_id = p.split('.')[0]
    p = str(frame_dir/p)
    save_root = save_dir
    # save_root = save_dir/c_name/v_id
    # save_root.mkdir(exist_ok=True, parents = True)
    
    f_name = p.split('/')[-1].split('.')[0]

    im = cv2.imread(str(p))
    # print("shape",im.shape)
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
    save_id = img_id +'.png'
    save_path = save_root/save_id
    # print(mask.shape,scores[i])
    img = im*alpha + (1-alpha)*overall_mask*255
    cv2.imwrite(str(save_path), img)        
    # break
    # cv2.imwrite(str(save_root/f_name), instance_result)

    # f_id = f_name.split('.')[0]
    # pkl_path = save_root/(f_id + '.pkl')
    # with open(str(pkl_path), 'wb') as handle:
    #     pickle.dump(c, handle, protocol=pickle.HIGHEST_PROTOCOL)


