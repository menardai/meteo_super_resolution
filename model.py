import fastai
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from fastai.vision import *
from fastai.callbacks import *

import resnet_group_norm as gn_models


def get_unet_resnet34_3_to_3(data_loader, weights_filename=None):
    """
    Returns a unet learner based on a Resnet-34 as encoder.
    The encoder takes as input and generate a 3 channels RGB PIL Image.
    """
    learn_gen = _create_gen_learner(data_loader, models.resnet34)

    if weights_filename is not None:
        learn_gen.load(weights_filename)

    return learn_gen


def get_unet_resnet34_gn_3_to_3(data_loader, weights_filename=None):
    """
    Returns a unet learner based on a Resnet-34 as encoder.
    The encoder takes as input and generate a 3 channels RGB PIL Image.
    """
    learn_gen = _create_gen_learner(data_loader, gn_models.resnet34)

    if weights_filename is not None:
        learn_gen.load(weights_filename)

    return learn_gen


def get_unet_resnet34_8_to_1(data_loader, weights_8_to_1_filename=None):
    """
    Returns a unet learner based on a Resnet-34 as encoder.
    The encoder takes as input a 8 channels 256x256 image and generate a 1 channel 256x256 image.
    """
    learn_gen = _create_gen_learner(data_loader, models.resnet34)

    model = learn_gen.model

    layer0_weight = model.layers[0][0].weight

    # Replace first convolutional layer 3->64 by 8->64
    model.layers[0][0] = nn.Conv2d(8,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)

    # keeping first 3 channels weights (pre-trained on imagenet)
    # and initializing weights of the additionnal 5 channels with zeros
    zero_new_channel = to_device(torch.zeros(64,5,7,7), data_loader.device)
    model.layers[0][0].weight = torch.nn.Parameter(torch.cat((layer0_weight, zero_new_channel), dim=1))

    # update model's layers 10 - from 99 channels to 104
    model.layers[10][0][0] = nn.Conv2d(104, 104, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model.layers[10][1][0] = nn.Conv2d(104, 104, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # update model's last layers (11) - from 99,3 channels to 104x1
    model.layers[11][0] = nn.Conv2d(104, 1, kernel_size=(1, 1), stride=(1, 1))

    if weights_8_to_1_filename is not None:
        learn_gen.load(weights_8_to_1_filename)

    # move to GPU after having modified the model weights
    learn_gen.model = to_device(model, data_loader.device)

    return learn_gen


def get_unet_resnet34_5_to_1(data_loader, weights_5_to_1_filename=None):
    """
    Returns a unet learner based on a Resnet-34 as encoder.
    The encoder takes as input a 5 channels 256x256 image and generate a 1 channel 256x256 image.
    """
    learn_gen = _create_gen_learner(data_loader, models.resnet34)

    model = learn_gen.model

    layer0_weight = model.layers[0][0].weight

    # Replace first convolutional layer 3->64 by 5->64
    model.layers[0][0] = nn.Conv2d(5,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)

    # keeping first 3 channels weights (pre-trained on imagenet)
    # and initializing weights of the additionnal 2 channels with zeros
    zero_new_channel = to_device(torch.zeros(64,2,7,7), data_loader.device)
    model.layers[0][0].weight = torch.nn.Parameter(torch.cat((layer0_weight, zero_new_channel), dim=1))

    # update model's layers 10 - from 99 channels to 101
    model.layers[10][0][0] = nn.Conv2d(101, 101, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model.layers[10][1][0] = nn.Conv2d(101, 101, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # update model's last layers (11) - from 99,3 channels to 101x1
    model.layers[11][0] = nn.Conv2d(101, 1, kernel_size=(1, 1), stride=(1, 1))

    if weights_5_to_1_filename is not None:
        learn_gen.load(weights_5_to_1_filename)

    # move to GPU after having modified the model weights
    learn_gen.model = to_device(model, data_loader.device)

    return learn_gen


def _create_gen_learner(data_loader, architecture):
    """
    Create a unet learner based on the specified encoder architecture using the given data loader.
    """
    wd = 1e-3  # weight decay
    y_range = (-3.,3.)
    loss_gen = MSELossFlat()

    return unet_learner(data_loader, architecture,
                        wd=wd, blur=True, norm_type=NormType.Weight,
                        self_attention=True, y_range=y_range, loss_func=loss_gen)



