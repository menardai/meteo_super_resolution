#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File: score.py
Author: Stephane Menard
Creation date: January 18th 2019
Description: Script that calculate the score of the tests images:
             - ./data/label_test_set.npy
             - ./images/image_gen.npy
Required package: numpy and tensorflow
'''

import numpy as np

from scoreFunction import *


lowres_test= np.load('./data/input_test_set.npy')
label_test= np.load('./data/label_test_set.npy')

#generated_test = np.load('./images/image_gen_test.npy')
#generated_test = np.load('./images/image_gen_5_1_test.npy')
generated_test = np.load('./images/image_gen_5_5p_test.npy')

print('lowres:', lowres_test.shape)
print('label : ', label_test.shape)
print('generated:', generated_test.shape)
print('------------------------------')

lowres_scores = getScore(lowres_test[:,5,:,:].reshape(248,1,256,256), label_test)
print('lowres score (mean) =', np.mean(lowres_scores))

generated_scores = getScore(generated_test, label_test)
print('generated score (mean) =', np.mean(generated_scores))

