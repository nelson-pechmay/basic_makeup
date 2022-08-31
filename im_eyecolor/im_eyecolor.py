# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:52:40 2022

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
hue_li = 183.
hue_ri = 31.
print( "USAGE : python im_lipstick.py -f file.jpg -a hue left-iris ( default : 183. ) -b hue right-iris ( default : 31. ) --saveIt ( default : omitted )" )

# input arguments:
ap = argparse.ArgumentParser()
ap.add_argument( "-f", "--filename", required=True, help="Path to the rgb image", type=str )
ap.add_argument( "-a", "--hueleft",  help="hue value for left iris default : 183.", type=float )
ap.add_argument( "-b", "--hueright",  help="hue value for right iris default : 31.", type=float )
ap.add_argument( "--saveIt", default=False, action='store_true', help="save output (default omitted), include to save the output", )

args = vars(ap.parse_args())

if(args["filename"]):
  filename = args[ "filename" ]
if(args["hueleft"]):
  hue_li = args["hueleft"]/2
if(args["hueright"]):
  hue_ri = args["hueright"]/2

# read color image:
src = cv2.imread( filename, cv2.IMREAD_COLOR )
src_x = cv2.resize( src, (800,800), interpolation=cv2.INTER_AREA )
rgb = cv2.cvtColor( src_x, cv2.COLOR_BGR2RGB )
hsv = cv2.cvtColor( src_x, cv2.COLOR_BGR2HSV )

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

# iris mask:
def iris_mask(im, p1, p2):
    
    _x0 = p1[0]-10
    _y0 = p1[1]-10
    _x1 = p2[0]+15
    _y1 = p2[1]+10
    _mask = np.zeros_like( im, dtype='uint8' )
    ##_mask[_y0:_y1,_x0:_x1] = 255
    _V = im[_y0:_y1,_x0:_x1,2] # s-channel (mask)
    _ret, _thresh = cv2.threshold( _V, 80, 255, 1 )
    _contours, _ = cv2.findContours( _thresh, cv2.RETR_LIST, 2 )
    _contours = max( _contours, key=cv2.contourArea )
    _elli = cv2.fitEllipse( _contours )
    (_xc, _yc), (_d1, _d2), _angle = _elli
    ##cv2.ellipse( _mask[_y0:_y1,_x0:_x1], _elli, (255,255,255), -2 )
    
    # draw smaller circle:
    r_minor = min( _d1, _d2 )/2
    cv2.circle( _mask[_y0:_y1,_x0:_x1], (int(_xc)+1,int(_yc)+3), int(1.01*r_minor), (255,255,255), -2 )
    
    # apply closing:
    # _element = cv2.getStructuringElement( cv2.MORPH_RECT, (2, 2) )
    # _mask = cv2.morphologyEx( _mask, cv2.MORPH_ERODE, _element, iterations=3 )
    # blur the above mask:
    _mask = cv2.GaussianBlur( _mask, (9, 9), 0 )
    
    return _mask

# color eyes:
def colored_eyes(im, li_mask, ri_mask, hue_li, hue_ri):
    # combined iris mask:
    both_mask = ri_mask | li_mask
    # apply eyes new color:
    src8 = np.copy( im )
    H, S, V = cv2.split( src8 )
    H[ ri_mask[:,:,0]>0 ] = hue_li
    H[ li_mask[:,:,0]>0 ] = hue_ri
    S[:,:] = 200
    #V[ both_mask[:,:,2]!=0 ] = 250
    src8 = cv2.merge( ( np.uint8(H), S, V ) )
    src8 = cv2.cvtColor( src8, cv2.COLOR_HSV2BGR )
    #
    maski32 = both_mask/255.
    src32 = src8/255.
    mask_eyes = cv2.multiply( src32, maski32 )
    
    return mask_eyes

# face mask:
def face_mask(im, mask):
    dst = np.copy( im )
    img32 = dst/255.
    inv_maski32 = mask/255.
    mask_face = cv2.multiply( img32, inv_maski32 )
    
    return mask_face

# left iris mask:
p38 = (points[37][0], points[37][1])
p41 = (points[40][0], points[40][1])
li_mask = iris_mask(hsv, p38, p41)
inv_li_mask = cv2.bitwise_not( li_mask )
# right iris mask:
p44 = (points[43][0], points[43][1])
p47 = (points[46][0], points[46][1])
ri_mask = iris_mask(hsv, p44, p47)
# inverse mask:
both_mask = ri_mask | li_mask
inv_both_mask = cv2.bitwise_not( both_mask )
# apply color to eyes mask:
mask_eyecolor = colored_eyes( hsv, li_mask, ri_mask, hue_li, hue_ri )
# make face mask:
mask_face = face_mask( src_x, inv_both_mask )
# 'blend' masks:
dst_eyescolor = mask_eyecolor + mask_face
dst_eyescolor = np.uint8( dst_eyescolor*255 )

# save output image:
if args["saveIt"]:
    shapex = ( src.shape[1], src.shape[0] )
    imgx = cv2.resize( dst_eyescolor, shapex, interpolation=cv2.INTER_AREA )
    cv2.imwrite( filename[:-4]+"_eyecolor.jpg", imgx )

# show images:
cv2.imshow( "Input", src )
cv2.imshow( "Mask", both_mask )
cv2.imshow( "Eyecolor Mask", mask_eyecolor )
cv2.imshow( "Output", dst_eyescolor )
cv2.waitKey(0)
cv2.destroyAllWindows()