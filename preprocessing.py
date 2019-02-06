import os
import glob

import numpy as np
from PIL import Image


def check_data_folder(data_dir):
    # make sure the data folder exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # look for some training or test data
    lowres_training_filenames, hires_training_filenames = get_data_filenames(data_dir, 'input_training_', 'label_training_')
    lowres_test_filenames, hires_test_filenames = get_data_filenames(data_dir, 'input_test_', 'label_test_')

    print('number of training files = {}'.format(len(lowres_training_filenames)))
    print('number of test files = {}'.format(len(lowres_test_filenames)))

    # make sure we have at least some training or test files
    if len(lowres_training_filenames) == 0 and len(lowres_test_filenames) == 0:
        print('*** Error - no training or test files ***')
        print('Please copy input_training_xxxx.npy and label_training_xxxx.npy in folder {}'.format(data_dir))
        return False

    if len(lowres_training_filenames) > 0:
        # make sure we have the same number of lowres/hires images
        if len(lowres_training_filenames) != len(hires_training_filenames):
            print('*** Error - not the same number of lowres/hires training files ***')
            return False

        # make sure the date file exists for training data
        if not os.path.isfile(os.path.join(data_dir, 'date_training.npy')):
            print('*** Error - cannot find file date_training.npy ***')

    if len(lowres_test_filenames) > 0:
        # make sure we have the same number of lowres/hires images
        if len(lowres_test_filenames) != len(hires_test_filenames):
            print('*** Error - not the same number of lowres/hires test files ***')
            return False

        # make sure the date file exists for test data
        if not os.path.isfile(os.path.join(data_dir, 'date_test_set.npy')):
            print('*** Error - cannot find file date_test_set.npy ***')

    return True


def get_data_filenames(data_dir, lowres_prefix, hires_prefix):
    """
        Returns a list of two arrays: lowres data filenames and hires data filenames.

        data_dir (string): the root data folder that contains lowres and hires data
        lowres_prefix (string): the filename prefix to seach for lowres data
        hires_prefix (string): the filename prefix to seach for hires data
    """
    lowres_filenames = sorted(glob.glob(os.path.join(data_dir, lowres_prefix + '*')))
    hires_filenames = sorted(glob.glob(os.path.join(data_dir, hires_prefix + '*')))

    return lowres_filenames, hires_filenames


def get_channel(vector_2d, min_value, max_value):
    """
    Returns a channel with values between [0, 255] of the same shape as the input vector_2d (usually 256x256).

    Arguments:
        vector_2d: (256,256) float vector; temperature ranging from [min_value, max_value].
        min_value: integer, the minimum value of the entire distribution of the given vector_2d
        max_value: integer, the maximum value of the entire distribution of the given vector_2d
    """
    norm_img = (vector_2d - min_value) / (max_value - min_value)

    norm_img[norm_img < 0] = 0   # clip negative numbers
    norm_img[norm_img > 1] = 1   # clip numbers over 1.0

    return (norm_img * 255).astype(np.uint8)


def get_rgb_images(lowres_sample, hires_sample):
    """
    Returns a PIL Image with green and blue channel all 0 and red channel temperature

    Arguments:
        lowres_sample: (15, 256, 256)
        hires_sample: (1, 256, 256)

    Returns:
        (PIL Image, PIL Image) : (lowres rgb image, hires rgb image)
    """
    lowres_rgb_channels = np.zeros((256,256,3)).astype(np.uint8)
    hires_rgb_channels  = np.zeros((256,256,3)).astype(np.uint8)

    wind_x_channel = get_channel(lowres_sample[13,:,:], -35, 40)
    wind_y_channel = get_channel(lowres_sample[14,:,:], -35, 40)

    lowres_rgb_channels[:,:,0] = get_channel(lowres_sample[5,:,:], -50, 50)
    lowres_rgb_channels[:,:,1] = wind_x_channel
    lowres_rgb_channels[:,:,2] = wind_y_channel

    hires_rgb_channels[:,:,0] = get_channel(hires_sample[0,:,:], -50, 50)
    hires_rgb_channels[:,:,1] = 0
    hires_rgb_channels[:,:,2] = 0

    return Image.fromarray(lowres_rgb_channels), Image.fromarray(hires_rgb_channels)


def export_png_images(lowres_npy_filenames, lowres_dest,
                      hires_npy_filenames, hires_dest,
                      dates):
    """
    Parse every sample from the specified .npy files and save a png image of the temperature.
    Each png file is named by its date.

    Arguments
        lowres_npy_filenames : array of string, list of lowres .npy files.
        lowres_dest : string, folder name where to save png image for lowres

        hires_npy_filenames : array of string, list of hires .npy files.
        hires_dest : string, folder name where to save png image for hires

        dates : date in chronological order of the given 'source_npy_filenames'
    """
    # create the destination folder
    if not os.path.exists(lowres_dest):
        os.makedirs(lowres_dest)
    if not os.path.exists(hires_dest):
        os.makedirs(hires_dest)

    date_index = 0

    for lowres_npy_filename, hires_npy_filename in zip(lowres_npy_filenames, hires_npy_filenames):
        lowres_samples = np.load(lowres_npy_filename)
        hires_samples  = np.load(hires_npy_filename)

        nb_samples = lowres_samples.shape[0]

        for i in range(nb_samples):
            # create a PIL Image of lowres and hires sample
            lowres_image, hires_image = get_rgb_images(lowres_samples[i,:,:,:], hires_samples[i,:,:,:])

            # save the PIL Images in dest folder using corresponding date as filename
            filename = dates[date_index].decode('UTF-8')+'.png'
            lowres_image.save(os.path.join(lowres_dest, filename), "PNG")
            hires_image.save(os.path.join(hires_dest, filename), "PNG")

            date_index += 1

            if date_index % 100 == 0:
                print('processed {}/{} images'.format(date_index, len(dates)))

    print('processed {}/{} images'.format(date_index, date_index))
    return date_index


if __name__ == "__main__":

    data_dir = './data'
    output_train_dir = './images/train'
    output_test_dir  = './images/valid'
    success = check_data_folder(data_dir)

    if success:
        print('PRE-PROCESSING TRAINING DATA')
        lowres_training_filenames, hires_training_filenames = get_data_filenames(data_dir, 'input_training_', 'label_training_')
        date_training = np.load(os.path.join(data_dir, 'date_training.npy'))

        # convert all input and label entries into png images
        export_png_images(lowres_training_filenames, os.path.join(output_train_dir, 'lowres'),
                          hires_training_filenames,  os.path.join(output_train_dir, 'hires'),
                          date_training)

        print('PRE-PROCESSING TEST DATA')
        lowres_test_filenames, hires_test_filenames = get_data_filenames(data_dir, 'input_test_', 'label_test_')
        date_test = np.load(os.path.join(data_dir, 'date_test_set.npy'))

        # convert all input and label entries into png images
        export_png_images(lowres_test_filenames, os.path.join(output_test_dir, 'lowres'),
                          hires_test_filenames,  os.path.join(output_test_dir, 'hires'),
                          date_test)
