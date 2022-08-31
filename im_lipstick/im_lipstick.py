# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:04:34 2022

@author: nelson w. pech-may
"""

import cv2,sys,dlib,argparse
import numpy as np
import matplotlib.pyplot as plt


# landmark model location:
PREDICTOR_PATH =  "../resources/shape_predictor_68_face_landmarks.dat"

# get the face detector:
faceDetector = dlib.get_frontal_face_detector()
# the landmark detector is implemented in the shape_predictor class:
landmarkDetector = dlib.shape_predictor( PREDICTOR_PATH )

# default inputs:
hue = 332.
print( "USAGE : python im_lipstick.py -f file.jpg -a hue value ( default : 332. ) --saveIt ( default : omitted )" )

# input arguments:
ap = argparse.ArgumentParser()
ap.add_argument( "-f", "--filename", required=True, help="Path to the rgb image", type=str )
ap.add_argument( "-a", "--hue",  help="hue value default : 332.", type=float )
ap.add_argument( "--saveIt", default=False, action='store_true', help="save output (default omitted), include to save the output" )

args = vars(ap.parse_args())

if(args["filename"]):
  filename = args[ "filename" ]
if(args["hue"]):
  hue = args["hue"]/2


# read rgb image:
src = cv2.imread( filename, cv2.IMREAD_COLOR )
src_x = cv2.resize( src, (800,800), interpolation=cv2.INTER_AREA )
rgb = cv2.cvtColor( src_x, cv2.COLOR_BGR2RGB )

# detect faces:
faces = faceDetector( rgb, 0 )
if len(faces):
    landmarks = landmarkDetector( rgb, faces[0] )
else:
    print( "No face detected in input image!" )

# read landmark points:
points = []
for i in range(68):
    point = (landmarks.part(i).x, landmarks.part(i).y)
    points.append( point )

# lips mask
def lips_mask(im):
    mask = np.zeros_like( im, dtype=np.float32 )
    cv2.fillPoly( mask, [np.int32(points[48:60])], color=(255,255,255), offset=(0,-2) )
    #inner_lip = cv2.fillConvexPoly( mask, np.int32(points[60:65]), 0 )
    # apply openning to the mask:
    _element = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (2, 3) )
    maski = cv2.morphologyEx( mask, cv2.MORPH_OPEN, _element, iterations=1 )
    # blur the above mask:
    maski = cv2.GaussianBlur( maski, (11, 7), 0 )
    maski = np.uint8( maski )
    
    return maski

# lipstick mask:
def lipstick_hue(im, mask, hue):
    #apply desired hue:
    src8 = np.copy( im )
    hsv = cv2.cvtColor( im, cv2.COLOR_BGR2HSV )
    H, S, V = cv2.split( hsv )
    H[:,:] = hue
    S[:,:] = 250
    src8 = cv2.merge( ( np.uint8(H), np.uint8(S), V ) )
    src8 = cv2.cvtColor( src8, cv2.COLOR_HSV2BGR )
    #
    maski32 = maski8/255.
    src32 = src8/255.
    mask_lips = cv2.multiply( src32, maski32 )
    return mask_lips

# face mask:
def face_mask(im, mask):
    dst = np.copy( im )
    img32 = dst/255.
    inv_maski32 = mask/255.
    mask_face = cv2.multiply( img32, inv_maski32 )
    
    return mask_face

# make lips mask:
maski8 = lips_mask( rgb )
# inverse mask:
inv_maski = cv2.bitwise_not( maski8 )
# apply lipstick:
mask_lips = lipstick_hue( src_x, maski8, hue )
# make face mask:
mask_face = face_mask( src_x, inv_maski )
# 'blend' masks:
dst_lipstick = mask_lips + mask_face
dst_lipstick = np.uint8( dst_lipstick*255 )

# save output image:
if args["saveIt"]:
    shapex = ( src.shape[1], src.shape[0] )
    imgx = cv2.resize( dst_lipstick, shapex, interpolation=cv2.INTER_AREA )
    cv2.imwrite( filename[:-4]+"_lipstick.jpg", imgx )

# show images:
cv2.imshow( "Input", src )
cv2.imshow( "Mask", maski8 )
cv2.imshow( "Lipstick Mask", mask_lips )
cv2.imshow( "Output", dst_lipstick )
cv2.waitKey(0)
cv2.destroyAllWindows()