import os 
from pathlib import Path
import pickle 
import random 
import pycocotools.mask as mask_util
import numpy as np 
import cv2
import json 
import pickle
import datumaro as dm
import logging
from detectron2.utils.logger import setup_logger

if __name__ == "__main__":
    # setup_logger(name="mask2former")

    logger = logging.getLogger(__name__)
    print("in main")
    def get_img_id(v_id, f_id):
        return str(v_id)+ '/'+str(f_id)
    ann_path = '/home/rmodi/crcv_work/stream_lined_occlusions/models/MOC-Detector/data/ucf24/UCF101v2-GT.pkl'

    #check the empty frames 
    with open(ann_path, 'rb') as fid:
        ann = pickle.load(fid, encoding='iso-8859-1')

    pred_pkl = '/home/rmodi/manural_annotations/Mask2Former/small_filtered_seg.pkl'
    with open(pred_pkl, 'rb') as fid:
        pred_pkl = pickle.load(fid)

    frame_dir = Path('/home/rmodi/crcv_work/stream_lined_occlusions/models/MOC-Detector/data/ucf24/rgb-images')

    l = ann['labels']
    l.append("NoAction")
    cls_to_label = {}
    for label, cls in enumerate(l):
        label = label+1
        cls_to_label[cls] = label 

    instance_id = 1 
    assigned_img_id = 1
    ass_id_to_v_id = {}
    #train split 
    split = 'train'
    v_ids = sorted(ann['train_videos'][0])
    iterable_list = []

    for done, v_id in enumerate(v_ids):
        print("done train", done, "/", len(v_ids))
        logger.info(" logger done train {:d}/{:d}".format(done,len(v_ids)))
        c = v_id.split('/')[0]
        n_frames = ann['nframes'][v_id]
        
        for f_id in range(1,n_frames+1):
            p = str(frame_dir/v_id/(str(f_id).zfill(5)+'.jpg'))
            img = cv2.imread(p)
            img_id = get_img_id(v_id,f_id)
            ass_id_to_v_id[assigned_img_id] = img_id
            label = cls_to_label[c]
            annotations = [] #the temp annotaitons which will go here 
            if f_id in pred_pkl[v_id].keys():
                rles = pred_pkl[v_id][f_id]
        
                for rle in rles:
                    mask = mask_util.decode(rle)
                    annotations.append(
                        dm.Mask(image=mask, label=label,
                        id= instance_id, attributes={'is_crowd': False})
                    )
                    instance_id+=1
                
                iterable_list.append(
                    dm.DatasetItem(
                        id = img_id,\
                        image = img,\
                        subset = split,\
                        attributes = {'id': img_id},\
                        annotations = annotations
                                )
                    )
            else:
                label = cls_to_label['NoAction']
                annotations =  []
                iterable_list.append(
                    dm.DatasetItem(
                        id = img_id,\
                        image = img,\
                        subset = split,\
                        attributes = {'id': img_id},\
                        annotations = annotations
                                )
                    )

    ###test split
    # split = 'test'
    # v_ids = sorted(ann['test_videos'][0])
    # iterable_list = []

    # for done, v_id in enumerate(v_ids):
    #     print("done test", done, "/", len(v_ids))
    #     logger.info("logger done test {:d}/{:d}".format(done,len(v_ids)))

    #     c = v_id.split('/')[0]
    #     n_frames = ann['nframes'][v_id]
        
    #     for f_id in range(1,n_frames+1):
    #         p = str(frame_dir/v_id/(str(f_id).zfill(5)+'.jpg'))
    #         img = cv2.imread(p)
    #         img_id = get_img_id(v_id,f_id)
    #         ass_id_to_v_id[assigned_img_id] = img_id
    #         label = cls_to_label[c]
    #         annotations = [] #the temp annotaitons which will go here 
    #         if f_id in pred_pkl[v_id].keys():
    #             rles = pred_pkl[v_id][f_id]
        
    #             for rle in rles:
    #                 mask = mask_util.decode(rle)
    #                 annotations.append(
    #                     dm.Mask(image=mask, label=label,
    #                     id= instance_id, attributes={'is_crowd': False})
    #                 )
    #                 instance_id+=1
                
    #             iterable_list.append(
    #                 dm.DatasetItem(
    #                     id = img_id,\
    #                     image = img,\
    #                     subset = split,\
    #                     attributes = {'id': img_id},\
    #                     annotations = annotations
    #                             )
    #                 )
    #         else:
    #             label = cls_to_label['NoAction']
    #             annotations =  []
    #             iterable_list.append(
    #                 dm.DatasetItem(
    #                     id = img_id,\
    #                     image = img,\
    #                     subset = split,\
    #                     attributes = {'id': img_id},\
    #                     annotations = annotations
    #                             )
    #                 )


    print("make dataset")
    dataset = dm.Dataset.from_iterable(iterable_list, categories=l)
    print("dumping dataset")
    dataset.export('./ucf_dataset', format='coco_instances',save_images=True)
