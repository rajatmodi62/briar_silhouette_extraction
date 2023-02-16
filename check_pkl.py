import os 
import numpy as np 
# from Pathlib import Path
import pickle 
import cv2
# dbfile = open('./data.pkl', 'rb')  
with open('./data.pkl', 'rb') as f:
    d = pickle.load(f)   
# db = pickle.load(dbfile)
# print(type(db))
print(d[0].shape)
cv2.imwrite('./mask.jpg', d[30]*255)