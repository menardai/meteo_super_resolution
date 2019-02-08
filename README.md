# Increasing Fine-Scale Temperature Details from Weather Model Forecasts

## Challenge
Increase the resolution (the level of detail) of 2D surface temperature forecasts obtained from Environment and 
Climate Change Canada (ECCC)’s weather forecast model, using as labelled images 2D temperature analysis at higher resolution. 
The scale factor between the model and the higher resolution analysis is 4 (from 10 km to 2.5 km). 

![lowres/hires](https://github.com/menardai/meteo_super_resolution/blob/master/challenge_doc/lowres-hires.jpg)  
 Here are examples of a temperature field over Western Canada at low resolution (10 km – left) and  
 high resolution (2.5 km – right).  The increase of resolution represents a factor of 4. 

## Proposed Solution

Using a computer vision super-resolution algorithm to increase the level of details. 
The U-NET architecture has proven its ability to perform image generation at a level that can compete with most GAN architectures.  

Below, the original U-NET:  
![unet](https://github.com/menardai/meteo_super_resolution/blob/master/challenge_doc/u-net-architecture.png)  

The proposed solution is a U-NET that sit on top of a Resnet-34 pretrained model (encoder).
The network takes as input a 3 channels 256x256 image and output of a 3 channels image.  

Input image:  
 - red channel: air temperature of the grid at low resolution (10km)
 - green channel: U-component of the wind along the X-axis of the grid
 - blue channel: V-component of the wind along the Y-axis of the grid
 
Target image (output):  
 - red channel: air temperature of the grid at high resolution (2.5 km)
 - green channel: 0
 - blue channel: 0

![generated image](https://github.com/menardai/meteo_super_resolution/blob/master/challenge_doc/input-rgb.png)  

Below, the same sample (date and time) but only the air temperature for the input, its target and the generate temperature produced by the model.  

![generated image](https://github.com/menardai/meteo_super_resolution/blob/master/challenge_doc/input-generated-target.png)  

## Requirements
 - 64-bit Python 3.6 installation. We recommend Anaconda3.
 - Pytorch 1.0
 - [Fastai library 1.0](https://github.com/fastai/fastai)

Run this command to install pytorch and fastai using Conda  
```conda install -c pytorch -c fastai fastai```  

## Preparing Dataset for Training
Convert the training and test dataset in RGB png images to be feed to the unet network.

First copy all the dataset files in a folder named *data* at project root and, then run the following script:  
```python preprocessing.py```  

Dataset from ./data/ folder will be converted and saved in png images in ./images/ folder with the following structure:  
<pre>
            ├── data  
            │   ├── date_test_set.npy  
            │   ├── date_training.npy  
            │   ├── date.txt  
            │   │  
            │   ├── input_test_set.npy  
            │   ├── label_test_set.npy  
            │   │   
            │   ├── input_training_0000_0099.npy  
            │   ├── input_training_0100_0199.npy  
            │   │     ...  
            │   ├── input_training_5200_5299.npy  
            │   ├── input_training_5300_5342.npy  
            │   │   
            │   ├── label_training_0000_0099.npy  
            │   ├── label_training_0100_0199.npy  
            │   │     ...  
            │   ├── label_training_5200_5299.npy  
            │   ├── label_training_5300_5342.npy  
            │  
            │  
            ├── images  
            │   ├── train  
            │   │   ├── hires  
            │   │   │   ├── 2016100100.png  
            │   │   │   └── 2016100103.png  
            │   │   └── lowres  
            │   │   │   ├── 2016100100.png  
            │   │   │   └── 2016100103.png  
            │   └── valid  
            │       ├── hires  
            │       │   ├── 2016100100.png  
            │       │   └── 2016100103.png  
            │       └── lowres  
            │           ├── 2016100100.png  
            │           └── 2016100103.png  
</pre>

## Training the Network
```python model_resnet34.py --train```  

Note that the dataset must have been preprocessed before training the model.  
(Please see the "Preparing Dataset for Training" section above)  

Training time for 200 epochs on a NVidia GTX 1070Ti 8Gb is about 10 hours.  


## Generating Increased-Resolution Temperature Images
```python model_resnet34.py --predict```  

The model to generate the images will be loaded from "./images/models/unet_resnet34.pth".  
The generated images will be saved in "./images/image_gen_test.npy".  




