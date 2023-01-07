rm_get_seg.py 
- was used to predict the instances from the maskformer2 model 

rm_get_framelevel_seg.py 
- for each frame, gets the detection bboxes needed 

rm_filter_instance_annotation.py 
- for all the preds, filters :
  a) scores > score_thresh and person class are kept 
  b) bbox drawn over instance. 
    - bbox iou with gt_iou should > iou_thresh 
  dumps filtered_seg.pkl to the disk. 

rm_create_datumoro. 
  - first step: convert ucf to coco for instance segmentation task. tubes for tracking can be built later  on. 
  - generates dataset in coco format for train/test split. 