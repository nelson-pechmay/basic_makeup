# basic_makeup
This repository is intented for the implementation of automated python scripts for applying basic makeup to images using computer vision tools.

# Description
This repository contains the implementation of the following makeup "filters":

1. im_lipstick
2. im_eyecolor

The scripts have been implemented using python, as part of the course Computer Vision II - Applications at opencv. The example images used in this repository are taken from the [lfw dataset](http://vis-www.cs.umass.edu/lfw/) and one image was provided by the course.

# Pre-requisites
Before running the scripts, please make sure to install in advance the following:
- python version 3.8
- opencv
- dlib for machine learning

# Usage of automated lipstick
In a terminal type: `python im_lipstick.py -f path_to_file -a hue_value -b True`

## Explanation:
- -f refers to the full path of the image you want to apply lipstick.
- -a refers to the *hue* value. This parameter controls the color of the lipstick.
- -b refers to the choice to save (True) or not (False) the image with the applied lipstick.

This script provides a very basic way to apply lipstick to a given image. It uses the HSV color model to do that. If you are familiar with the RGB color model, then you can think of HSV as the "cylindrical coordinate system" for color, being RGB, the "cartesian coordinate system". **Hue** gives the "sense" of color in human vision. Further explanation about the HSV color model can be found in the [Wikipedia](https://en.wikipedia.org/wiki/HSL_and_HSV) and a good resource to find the actual value of *hue* for a desired lipstick color can be found [here](https://www.color-hex.com/).

One of the main limitations of this implementation is that it assumes the lips are closed in the input image. Therefore, it will fail to "nicely" apply the lipstick if the lips are wide open in the given image.

# Usage of automated eye-color

# Sample results
Input image                   |  Lipstick applied
:----------------------------:|:--------------------------------------:
![](data/girl-no-makeup.jpg)  |  ![](data/girl-no-makeup_lipstick.jpg)
# TODOs
Here some todo list

