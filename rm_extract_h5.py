#ccode to extract the h5 file 
# apparently that is only the thing which is remaining 

import os 
from pathlib import Path 
import torch
import h5py

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