#code to increase the frame rate
#batched forward pass 
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
import time
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True


from mask2former.maskformer_model import MaskFormer

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

model = build_model(cfg).cuda()
model.eval()
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)
aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

input_format = cfg.INPUT.FORMAT

batch_size = 70
x = np.zeros((batch_size,224,224,3))
with torch.no_grad(): 

    if input_format == "RGB":
        # whether the model expects BGR inputs or RGB
        print(x.shape)
        x = x[:,:,:,::-1] 
        aug_images = []
        n_images = x.shape[0]
        for img_id in range(n_images):
            img = x[img_id]
            h_orig, w_orig,_ = img.shape
            print("image shape", img.shape)
            img = aug.get_transform(img).apply_image(img)
            aug_images.append(img)
        x = np.stack(aug_images, 0)
        x =  torch.as_tensor(x.astype("float32")).cuda()
        x = rearrange(x, 'b h w c-> b c h w')
        print(h_orig, w_orig)

        input = []
        for img_id in range(n_images):
            d = {"image": x[img_id], "height": h_orig, "width": w_orig}
            input.append(d)

        tic = time.time()
        predictions = model(input)
        toc = time.time()
        print(toc-tic,tic, toc)
        time_per_image = (toc-tic)/batch_size
        fps = 1/time_per_image
        print("fps",fps)
        print("type of", type(predictions), len(predictions))