import os 
from pathlib import Path
import pickle 
import random 
import pycocotools.mask as mask_util
import numpy as np 
from einops import rearrange, reduce, repeat
import cv2
import json 
import pickle

def iou_mask(m1, m2):
    intersection = np.sum(m1+m2==2)
    union = 1e-7 + np.sum(m1) + np.sum(m2)
    return intersection/union

def generate_rectangle_coors(mask):
  l_i, l_j = float('inf'), float('inf')
  r_i, r_j = -float('inf'), -float('inf')
  if np.sum(mask) ==0 :
    return None
  points = np.nonzero(mask)
  i_coors , j_coors = points[0], points[1]
  n = i_coors.shape[0]
  for idx in range(n):
    i = i_coors[idx]
    j = j_coors[idx]
    l_i, l_j = min(l_i, i), min(l_j,j)
    r_i, r_j = max(r_i, i), max(r_j,j)
  return l_i, l_j, r_i, r_j

ann_path = '/home/rmodi/crcv_work/stream_lined_occlusions/models/MOC-Detector/data/ucf24/UCF101v2-GT.pkl'

#read frame wise annotation 
gt_json = './seg.json' #contains the vid along with the f_id
score_thresh = 0.3
iou_thresh = 0.2 # pred box should intersect at least one gt  box with this iouy
v_id = 'SalsaSpin/v_SalsaSpin_g10_c04'
f_id = 20

f = open(gt_json)
gt_json = json.load(f)



#check the empty frames 
with open(ann_path, 'rb') as fid:
    ann = pickle.load(fid, encoding='iso-8859-1')

v_ids = ann['train_videos'][0] + ann['test_videos'][0]

pkl_dir = Path('/home/rmodi/manural_annotations/Mask2Former/maskformer_outputs')

d = {} # to save
for done, v_id in enumerate(sorted(v_ids)):
    print("done", done, "/", len(v_ids))
    d[v_id] = {}
    try:
        n_frames = ann['nframes'][v_id]
        for f_id in range(1, n_frames+1):
            d[v_id][f_id] = []
            pred_path = pkl_dir/v_id/(str(f_id).zfill(5)+'.pkl')
            with open(str(pred_path), 'rb') as fid:
                pred = pickle.load(fid)

            # print(pred.keys())
            boxes = pred['boxes']
            classes = pred['classes']
            scores = pred['scores']
            rle = pred['rle']

            masks = []
            # boxes = []

            if str(f_id) in gt_json[v_id].keys():

                gt_boxes = gt_json[v_id][str(f_id)]

                for i,r in enumerate(rle):
                    if classes[i]==0 and scores[i]>score_thresh:

                        # print("scoores", scores[i])
                        mask = mask_util.decode(r)
                        
                        y1,x1,y2,x2 = generate_rectangle_coors(mask)
                        pred_mask = np.zeros((240,320))
                        pred_mask[y1:y2, x1:x2] = 1
                        masks.append([pred_mask, mask]) #[rectangle mask, instance_mask]

                        # p = str(i)+'.png'
                        # cv2.imwrite(p, 255*pred_mask)
                        # p = str(i)+'_g.png'
                        # cv2.imwrite(p, 255*mask)
                        # print(mask.shape,np.unique(mask))

                kept_mask = []
                for id, mask in enumerate(masks):
                    pred_rec, pred_instance = mask[0], mask[1]
                    curr_iou = -float('inf')
                    for gt_id, gt_box in enumerate(gt_boxes):
                        x1,y1,x2,y2 = gt_box
                        gt_mask = np.zeros((240,320))
                        gt_mask[y1:y2,x1:x2] = 1
                        iou = iou_mask(pred_rec, gt_mask)
                        # print("iou is", iou)
                        curr_iou = max(curr_iou, iou)

                        # p = str(gt_id) +'_gt.png'
                        # cv2.imwrite(p, 255*gt_mask)
                    # print("curr iou", curr_iou)
                    if curr_iou>=iou_thresh:
                        kept_mask.append(mask)

                for id, mask in enumerate(kept_mask):
                    pred_rec, pred_instance = mask[0], mask[1]
                    encoded_rle = mask_util.encode(np.array(pred_instance, order="F", dtype="uint8"))
                    d[v_id][f_id].append(encoded_rle)

                    p = Path('/home/rmodi/manural_annotations/Mask2Former/filtered_outputs')/v_id.split('/')[-1]
                    p.mkdir(exist_ok=True, parents=True)
                    p = p/(v_id.split('/')[-1]+'_'+str(f_id)+'_.pkl')
                    f = open(str(p), 'wb')
                    pickle.dump(encoded_rle, f)
                    # p = str(id)+'matched_.png'
                    # cv2.imwrite(p, 255*pred_instance)
                # print(v_id)
                # print(d)
                # exit(1)
    except:
        print("some error")

print(d)
f = open('small_filtered_seg.pkl', 'wb')
pickle.dump(d, f)
