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
save = False
print( "USAGE : python im_lipstick.py -f file.jpg -a hue value ( default : 332. ) -b save output? ( default : False )" )

# input arguments:
ap = argparse.ArgumentParser()
ap.add_argument( "-f", "--filename", required=True, help="Path to the rgb image", type=str )
ap.add_argument( "-a", "--hue",  help="hue value default : 170.", type=float )
ap.add_argument( "-b", "--save",  help="save output default : False", type=bool )

args = vars(ap.parse_args())

if(args["filename"]):
  filename = args[ "filename" ]
if(args["hue"]):
  hue = args["hue"]/2
if(args["hue"]):
  save = args["save"]


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

# make lips mask:
mask = np.zeros_like( rgb, dtype=np.float32 )
outer_lip = cv2.fillPoly( mask, [np.int32(points[48:60])], color=(255,255,255), offset=(0,-4) )
#inner_lip = cv2.fillConvexPoly( mask, np.int32(points[60:65]), 0 )
# apply openning to the mask:
_element = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (3, 4) )
maski = cv2.morphologyEx( mask, cv2.MORPH_ERODE, _element, iterations=3 )
# blur the above mask:
maski = cv2.GaussianBlur( maski, (9, 5), 0 )
maski = np.uint8( maski )

# inverse mask:
inv_maski = cv2.bitwise_not( maski )

# apply lipstick:
src01 = np.copy( src_x )
hsv = cv2.cvtColor( src_x, cv2.COLOR_BGR2HSV )
H, S, V = cv2.split( hsv )
H[ maski[:,:,0]!=0 ] = hue
S[ maski[:,:,1]!=0 ] = 250
#V[ maski[:,:,2]!=0 ] = 96
src01 = cv2.merge( ( np.uint8(H), S, V ) )
src01 = cv2.cvtColor( src01, cv2.COLOR_HSV2BGR )
src01[maski==0] = maski[maski==0]

# lipstick source image:
dst_lipstick = np.copy( src_x )
dst_lipstick[src01!=0] = src01[src01!=0]

# save output image:
if save:
    shapex = ( src.shape[1], src.shape[0] )
    imgx = cv2.resize( dst_lipstick, shapex, interpolation=cv2.INTER_AREA )
    cv2.imwrite( filename[:-4]+"_lipstick.jpg", imgx )

# show images:
cv2.imshow( "Input", src )
cv2.imshow( "Mask", maski )
cv2.imshow( "Lipstick Mask", src01 )
cv2.imshow( "Output", dst_lipstick )
cv2.waitKey(0)
cv2.destroyAllWindows()
