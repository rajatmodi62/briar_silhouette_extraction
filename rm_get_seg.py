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
coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")
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

frame_dir = Path('/home/rmodi/crcv_work/stream_lined_occlusions/models/MOC-Detector/data/ucf24/rgb-images')

frame_paths = []
class_names = sorted(os.listdir(frame_dir))

for c in class_names:
    v_ids = sorted(os.listdir(str(frame_dir/c)))
    for v_id in v_ids:
        root = frame_dir/c/v_id
        f_ids = sorted(os.listdir(str(root)))
        for f in f_ids:
            p = root/f
            frame_paths.append(p)

            # print(frame_paths)

save_dir = Path('/home/rmodi/manural_annotations/Mask2Former/maskformer_outputs')
# random.shuffle(frame_paths)
for done, p in enumerate(frame_paths):
    print("done",done+1,"/", len(frame_paths))
    v_id = str(p).split('/')[-2]
    c_name = v_id.split('_')[1]
    f_name = str(p).split('/')[-1]
    save_root = save_dir/c_name/v_id
    save_root.mkdir(exist_ok=True, parents = True)

    im = cv2.imread(str(p))
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
    rm_outputs = outputs['instances'].to("cpu")
    # print("type of outputs", type(rm_outputs))
    boxes = rm_outputs.pred_boxes
    scores = rm_outputs.scores
    classes = rm_outputs.pred_classes
    masks =rm_outputs.pred_masks
    masks = rearrange(masks, 'n h w -> h w n')

    # print("mask shape", masks.shape,np.unique(masks))
    rle = mask_util.encode(np.array(masks, order="F", dtype="uint8"))
    # new_mask = mask_util.decode(rle)
    # print(np.array_equal(masks, new_mask))
    # print("len of masks", len(masks))
    c = {'boxes':boxes,\
        'scores':scores,\
         'classes':classes,\
          'rle':rle
    }

    # continue
    # exit(1)
    cv2.imwrite(str(save_root/f_name), instance_result)

    f_id = f_name.split('.')[0]
    pkl_path = save_root/(f_id + '.pkl')
    with open(str(pkl_path), 'wb') as handle:
        pickle.dump(c, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # exit(1)
# im = cv2.imread("./input.jpg")


# outputs = predictor(im)

# print("outputs", outputs.keys(),type(outputs['instances']))
# v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
# instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
# cv2.imwrite('./out.jpg', instance_result)