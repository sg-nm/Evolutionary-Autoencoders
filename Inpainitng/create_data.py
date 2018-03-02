# -*- coding: utf-8 -*-
#!/usr/bin/env python

import cv2
import os
import random
import numpy as np


p_size = 48
data_dir = "./celebA_org/"
out_dir1 = "./dataset/celebA/train/train/"
out_dir2 = "./dataset/celebA/val/val/"
out_dir3 = "./dataset/celebA/test/test/"
if not os.path.exists(out_dir1):
    os.makedirs(out_dir1)
if not os.path.exists(out_dir2):
    os.makedirs(out_dir2)
if not os.path.exists(out_dir3):
    os.makedirs(out_dir3)

fp = open('list_landmarks_align_celeba.txt', 'r')
line = fp.readline().strip()
line = fp.readline().strip()
count = 0
sumR = 0.0
sumG = 0.0
sumB = 0.0
patchSize = 64
while line != '':
    line = fp.readline().strip()
    key = line.split()
    x = int(key[5])  # nose_x
    y = int(key[6])  # nose_y
    x1 = int(key[1]) # Leye_x
    y1 = int(key[2]) # Leye_y
    x2 = int(key[3]) # Reye_x
    y2 = int(key[4]) # Reye_y
    center_eye_X = int((x2+x1)/2.0)
    center_eye_Y = int((y2+y1)/2.0)
    img = cv2.imread(data_dir+key[0],1)
    patch = img[center_eye_Y-p_size:center_eye_Y+p_size, center_eye_X-p_size:center_eye_X+p_size, :]
    out = cv2.resize(patch, (patchSize,patchSize), interpolation=cv2.INTER_CUBIC)
    if count < 100000:
        a = 1
        # cv2.imwrite(out_dir1+'train/train/'+str("{0:05d}".format(count))+'.png', out)
    elif count < 101000:
        cv2.imwrite(out_dir2+str("{0:05d}".format(count))+'.png', out)
    elif count < 103000:
        cv2.imwrite(out_dir3+str("{0:05d}".format(count))+'.png', out)        
    count += 1

