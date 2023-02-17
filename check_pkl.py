import os 
import numpy as np 
# from Pathlib import Path
import pickle 
import cv2
# dbfile = open('./data.pkl', 'rb')  
# with open('./data.pkl', 'rb') as f:
#     d = pickle.load(f)   
# # db = pickle.load(dbfile)
# # print(type(db))
# print(d[0].shape)
# cv2.imwrite('./mask.jpg', d[30]*255)
import torch 
mask = torch.load('./mask.torch',map_location=torch.device('cpu'))

f_id = 25 
x = mask[f_id].numpy()
x = np.expand_dims(x,2)
x = (x==0)*1
print(np.unique(x))
cv2.imwrite('./qw.jpg', np.uint8(255*x))