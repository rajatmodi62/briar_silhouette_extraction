#ccode to extract the h5 file 
# apparently that is only the thing which is remaining 

import os 
from pathlib import Path 
import torch
import h5py

p =  '/data/project/clip_extraction_brs_bts_v2/afs_brs1_static/brs_bts_version_1/BRS1/G00021/field/500m/wb/G00021_set1_struct_1623185822092_45efd745/frame_000000_000039_clip_0000/clip.h5'

clip_data = h5py.File(p, 'r')
data = clip_data['data']

print(data.shape)