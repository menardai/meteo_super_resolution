#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
File: score.py
Author: Stephane Menard
Creation date: January 18th 2019
Description: Script that calculate the score of the tests images:
             - ./data/label_test_set.npy
             - ./images/image_gen.npy

             Output the score for each image in ./images/image_gen_test_score.csv

** Required package: numpy and tensorflow **
'''

import os
import numpy as np

from scoreFunction import *


def compute_scores(generated_test, label_test, verbose=False):
    """
    Compute the score for each image.

    Args:
        generated_test: list (n,1,256,256) of generated test image
        label_test: list (n,1,256,256) of the label test image
        verbose: print score if verbose is True
    Return:
        a list of score
    """
    generated_scores = []

    img1_placeholder = tf.placeholder(tf.float32, shape = (None, 1, 256, 256), name = "img1_placeholder")
    img2_placeholder = tf.placeholder(tf.float32, shape = (None, 1, 256, 256), name = "img2_placeholder")
    score = scoreCalculation(img1_placeholder, img2_placeholder, 1.0)

    with tf.Session() as sess:

        for i, gen_img, label_img in zip(range(len(generated_test)), generated_test, label_test):

            scoreValue = score.eval(feed_dict = {img1_placeholder: gen_img.reshape(1,1,256,256),
                                                 img2_placeholder: label_img.reshape(1,1,256,256)})

            if verbose:
                print('{} - {:0.4f}'.format(i, scoreValue.item()))

            generated_scores.append(scoreValue.item())

    return generated_scores


def save_scores(generated_scores, date_times, output_csv_filename, verbose=False):
    with open(output_csv_filename, 'w') as f:
        for i in range(len(date_times)):
            date_time = date_times[i].decode('UTF-8')
            score = generated_scores[i]

            if verbose:
                print('{} --> {:0.4f}'.format(date_time, score))

            f.write('"{}","{:0.4f}"\n'.format(date_time, score))


if __name__ == "__main__":
    date_times = np.load('./data/date_test_set.npy')
    label_test = np.load('./data/label_test_set.npy')

    generated_test = np.load('./images/image_gen_test.npy')

    print('label : ', label_test.shape)
    print('generated:', generated_test.shape)
    print('------------------------------')

    output_csv_filename = './images/image_gen_test_score.csv'
    generated_scores = compute_scores(generated_test, label_test, verbose=True)
    save_scores(generated_scores, date_times, output_csv_filename, verbose=True)

    print('generated score (mean) =', np.mean(generated_scores))
    print('\nThe score for each image has been saved in this file:\n{}'.format(output_csv_filename))

    # ----------------------
    # compute score for baseline (lowres images against label images)
    # lowres score (mean) = 10.5384
    #lowres_test= np.load('./data/input_test_set.npy')
    #lowres_scores = compute_scores(lowres_test[:,5,:,:].reshape(248,1,256,256), label_test, verbose=True)
    #print('lowres score (mean) =', np.mean(lowres_scores))

