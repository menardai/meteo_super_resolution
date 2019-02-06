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

![generated image](https://github.com/menardai/meteo_super_resolution/blob/master/challenge_doc/input-generated-target-air.png)  


