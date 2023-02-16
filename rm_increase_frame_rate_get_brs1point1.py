#code to increase the frame rate
#batched forward pass 
import os 
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")
import h5py
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
torch.cuda.init()
cfg = get_cfg()
cfg.INPUT.MIN_SIZE_TEST = 512
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False


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
root_dir = Path('/home/data-rawat/crops_version_v1/static_crops/brs1.1')
save_root = Path('/home/data-rawat/godzilla_silhouette')

filtered_f_ids = []
with open('to_extract.txt') as f:
    lines = f.readlines()

for line in lines:
     p = line.split(' ')[0]
     filtered_f_ids.append([p,'clip.h5'])
    #  print(filtered_f_ids)
    #  exit(1)
# for p in os.walk(str(root_dir)):
#     v_folder = p[0]
#     # v_folder = v_folder.split('/')[2:]
#     # v_folder = '/'.join(v_folder)

#     f_ids = p[2]
    
#     for f_id in f_ids:
#         if '.h5' in f_id:
#             filtered_f_ids.append([v_folder,f_id])
#             # print("true")
print("len of", len(filtered_f_ids),filtered_f_ids[0])


batch_size = 64 
score_thresh = 0.5
for done, item in enumerate(filtered_f_ids):
    # print("done", done, "/", len(filtered_f_ids))
    
    try:
        v_folder, h5_path = item
        src_path = Path(v_folder)/h5_path
        # print("v_folder",v_folder)
        # exit(1)
        clip_data = h5py.File(str(src_path), 'r')
        data = clip_data['data']
        n_frames = data.shape[0]
        n_batches = n_frames//batch_size 
        if n_frames%batch_size!=0:
            n_batches+=1
        save_data = []
        for i in range(n_batches):
            print("done", done, "/", len(filtered_f_ids),i+1,"/",n_batches)
            # print(batch_size*i, batch_size*(i+1))
            start = batch_size*i
            end = batch_size*(i+1)
            x= data[batch_size*i:batch_size*(i+1)]
            with torch.no_grad():
                x = x[:,:,:,::-1]
                n_el = x.shape[0]
                # for offset in range(n_el):
                #     f_id = start + offset 
                #     print("f_id",f_id,n_frames)
                aug_images = []
                n_images = x.shape[0]
                for img_id in range(n_images):
                    img = x[img_id]
                    h_orig, w_orig,_ = img.shape
                    # print("image shape", img.shape)
                    img = aug.get_transform(img).apply_image(img)
                    aug_images.append(img)
                x = np.stack(aug_images, 0)
                x =  torch.as_tensor(x.astype("float32")).cuda()
                x = rearrange(x, 'b h w c-> b c h w')
                # print(h_orig, w_orig)

                input = []
                for img_id in range(n_images):
                    d = {"image": x[img_id], "height": h_orig, "width": w_orig}
                    input.append(d)

                tic = time.time()
                predictions = model(input)
                toc = time.time()

                
                for img_id in range(n_images):
                    

                    f_id = start + img_id
                    # print("f_id",f_id)
                    #choose the current pred
                    pred = predictions[img_id]
                    
                    print("rm_oututs",pred.keys())
                    # exit(1)
                    rm_outputs = pred['sem_seg'].to('cpu').numpy().argmax(0)
                    


                    # boxes = rm_outputs.pred_boxes
                    # scores = rm_outputs.scores
                    # classes = rm_outputs.pred_classes
                    # masks =rm_outputs.pred_masks
                    # masks = rearrange(masks, 'n h w -> h w n')
                    # n_preds = classes.shape[0]
                    # overall_mask = None
                    # print("no of preds",n_preds)
                    # exit(1)
                    # for i in range(n_preds):
                    #     c = classes[i] 
                    #     if c==0:
                    #         mask = masks[:,:,i].numpy()
                    #         mask = repeat(mask, 'h w -> h w c', c = 1)
                    #         if scores[i] > score_thresh:
                    #             # print(mask.shape, overall_mask.shape)
                    #             if overall_mask is None:
                    #                 overall_mask = mask
                    #             else:
                    #                 overall_mask+=mask#.astype(np.float)
                    overall_mask = (rm_outputs==0)*1
                    # print("overall mask shape",overall_mask.shape)
                    # print("before", save_root)
                    # print("v_folder before split", v_folder)
                    new_v_folder = str(v_folder).split('/')[2:]
                    
                    new_v_folder = '/'.join(new_v_folder)
                    dest_path = Path(save_root)/new_v_folder
                    # print("dest path",dest_path)
                    # print("v_folder",new_v_folder)
                    # print("overall mask",overall_mask.shape, type(overall_mask))
                    dest_path.mkdir(exist_ok=True, parents =True)
                    dest_path = dest_path/(str(f_id) + '.jpg')
                    # print(dest_path)
                    # print('---------------------------')
                    overall_mask = np.expand_dims(overall_mask,2)
                    # exit(1)
                    # cv2.imwrite(str(dest_path), np.uint8(overall_mask*255))
                    save_data.append(overall_mask)
                    # print("written")
                # exit(1)
        dest_path = Path(save_root)/new_v_folder/'data.pkl'
        dbfile = open(str(dest_path), 'ab')
      
        # source, destination
        # print("doing")
        pickle.dump(save_data, dbfile)                     
        dbfile.close()
        # exit(1)
    except:
        print("some error occured ")