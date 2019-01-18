#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File: scoreFunction.py
Author: Kevin Gauthier
Creation date: November 11th 2018
Description: Script that calculate the score of the tests images. (function getScore)
Required package: numpy and tensorflow
'''
import numpy as np
import tensorflow as tf


'''
  * Calculate the difference between the mean of two images
  * @param {img1|img2} : The two images
  * @returns {tf.abs(img1_mean - img2_mean)} Returns the difference between the mean values
'''
def meanDifferenceCalculation(img1, img2):
    img1_mean = tf.reduce_mean(img1)
    img2_mean = tf.reduce_mean(img2)

    return tf.abs(img1_mean - img2_mean)

'''
  * Calculate the difference between the min of two images
  * @param {img1|img2} : The two images
  * @returns {tf.abs(img1_min - img2_min)} Returns the difference between the min values
'''
def minDifferenceCalculation(img1, img2):
    img1_min = tf.reduce_min(img1)
    img2_min = tf.reduce_min(img2)

    return tf.abs(img1_min - img2_min)

'''
  * Calculate the difference between the max of two images
  * @param {img1|img2} : The two images
  * @returns {tf.abs(img1_max - img2_max)} Returns the difference between the max values
'''
def maxDifferenceCalculation(img1, img2):
    img1_max = tf.reduce_max(img1)
    img2_max = tf.reduce_max(img2)

    return tf.abs(img1_max - img2_max)

'''
  * Calculate the score between two images
  * @param {img1|img2|maxValue} : The two tensor images and the max values of a pixel in the two images
  * @returns {score} Returns the difference between the mean values
'''
def scoreCalculation(img1, img2, maxValue):
    #Transpose the tensors to have format (NHWC) / numpy array format given is (NCHW)
    img1_transpose = tf.transpose(img1, [0, 2, 3, 1])
    img2_transpose = tf.transpose(img2, [0, 2, 3, 1])

    #Calculation of all the intermediate score
    ssim = tf.image.ssim(img1_transpose, img2_transpose, max_val = maxValue)
    mae = tf.reduce_sum(tf.abs(tf.subtract(img1_transpose, img2_transpose))) / 256 / 256
    meanDifference = meanDifferenceCalculation(img1_transpose, img2_transpose)
    minDifference = minDifferenceCalculation(img1_transpose, img2_transpose)
    maxDifference = maxDifferenceCalculation(img1_transpose, img2_transpose)

    #Score function
    score = (2 - 2 * ssim) + mae + meanDifference + minDifference + maxDifference

    return score

'''
  * Calculate the score between two images using tensorflow session run
  * @param {img1|img2|maxValue} : The two tensor images and the max values of a pixel in the two images (default value is 1.0 for normalized images)
  * @returns {scoreValue} Returns a numpy array of the score value of each images
'''
def getScore(img1, img2, maxValue = 1.0):
    img1_placeholder = tf.placeholder(tf.float32, shape = (None, 1, 256, 256), name = "img1_placeholder")
    img2_placeholder = tf.placeholder(tf.float32, shape = (None, 1, 256, 256), name = "img2_placeholder")

    with tf.Session() as sess:
        score = scoreCalculation(img1_placeholder, img2_placeholder, maxValue)
        scoreValue = score.eval(feed_dict = {img1_placeholder: img1, img2_placeholder: img2})

        return scoreValue