# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 00:59:14 2019
@author: Rishika Parashar 
"""
import numpy as np
import sys
import cv2
import math
"""
w1 = 0.3
h1 = 0.3
w2 = 0.7
h2 = 0.7
inImage_name = 'singlecolor.png'
#inImage_name = 'stunningmesh.jpg'
#inImage_name = 'good-test-image-for-proj1.bmp'
#inImage_name = 'fruits.jpg'
outImage_name = 'output.png'
"""
if(len(sys.argv) != 7) :
    print(sys.argv[0], ": take 6 arguments . Not ", len(sys.argv)-1)
    print("Expecting arguments w1,h1,w2,h2,image_input and image_output as input arguments")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
inImage_name = sys.argv[5]
outImage_name = sys.argv[6]

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1):
  print("arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
  sys.exit()
  
inputImage = cv2.imread(inImage_name, cv2.IMREAD_COLOR)
if inputImage is None:
  print(": Failed to read image from ", inImage_name)
  sys.exit()
  
cv2.imshow("input image :" + inImage_name, inputImage)

rows, cols, bands = inputImage.shape
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

Y_min = float('inf')
Y_max = -float('inf')
Xw, Yw, Zw = [0.95, 1.0, 1.09]
uw = (4*Xw)/(Xw+(15*Yw)+(3*Zw))
vw = (9*Yw)/(Xw+(15*Yw)+(3*Zw))
A = 0
B = 1

# To get the minimum and maximum Y values
for i in range(H1, H2+1):
    for j in range(W1, W2+1):
        b, g, r = inputImage[i, j]

        if b > 255:
            b = 255
        elif b < 0:
            b = 0
        if g > 255:
            g = 255
        elif g < 0:
            g = 0
        if r > 255:
            r = 255
        elif r < 0:
            r = 0

        #Converting from sRGB to non-linear RGB 
        b = b/255
        g = g/255
        r = r/255

        #Converting from non-linear RGB to linear RGB
        if(r < 0.03928):
            r = r / 12.92
        else :
            r = pow(((r + 0.055) / 1.055), 2.4)
        if (g < 0.03928):
            g = g / 12.92
        else:
            g = pow(((g + 0.055) / 1.055), 2.4)
        if (b < 0.03928):
            b = b / 12.92
        else:
            b = pow(((b + 0.055) / 1.055), 2.4)

        #Converting from linear RGB to XYZ
        X = ((0.412453 * r) + (0.35758 * g) + (0.180423 * b))
        Y = ((0.212671 * r) + (0.71516 * g) + (0.072169 * b))
        Z = ((0.019334 * r) + (0.119193 * g) + (0.950227 * b))

        #Maximum and Minimum values of Y
        if (Y > Y_max):
            Y_max = Y
        if (Y < Y_min):
            Y_min = Y
            
# linear scaling of Y values and convert the image to back to sRGB.
for i in range(H1,H2+1):
    for j in range(W1,W2+1):
        b, g, r = inputImage[i, j]

        if b > 255:
            b = 255
        elif b < 0:
            b = 0
        if g > 255:
            g = 255
        elif g < 0:
            g = 0
        if r > 255:
            r = 255
        elif r < 0:
            r = 0

        # Converting from sRGB to non-linear RGB
        b = b / 255
        g = g / 255
        r = r / 255

        # Converting from non-linear RGB to linear RGB
        if (r < 0.03928):
            r = r / 12.92
        else:
            r = pow(((r + 0.055) / 1.055), 2.4)
        if (g < 0.03928):
            g = g / 12.92
        else:
            g = pow(((g + 0.055) / 1.055), 2.4)
        if (b < 0.03928):
            b = b / 12.92
        else:
            b = pow(((b + 0.055) / 1.055), 2.4)

        # Converting from linear RGB to XYZ
        X = ((0.412453 * r) + (0.35758 * g) + (0.180423 * b))
        Y = ((0.212671 * r) + (0.71516 * g) + (0.072169 * b))
        Z = ((0.019334 * r) + (0.119193 * g) + (0.950227 * b))

        #Calculating the x and y values after scaling the Y values
        if X == 0 and Y == 0 and Z == 0:
            x = 0.3127
            y = 0.3291
        else:
            x = X/(X+Y+Z)
            y = Y/(X+Y+Z)
        
        #Linear scaling of Y values in the range [A,B]
        if Y > Y_max:
            Y = 1
        elif Y < Y_min:
            Y = 0
        else:
            Y = (Y - Y_min) * ((B - A) / (Y_max - Y_min))
        
        #Converting from xyY to XYZ domain 
        if y == 0:
            X = 0
            Y = 0
            Z = 0
        else:
            X = x * Y / y
            Y = Y
            Z = (1-x-y) * Y / y

        #Converting from XYZ to linear sRGB
        r = ((3.240479 * X) + (-1.53715 * Y) + (-0.498535 * Z))
        g = ((-0.969256 * X) + (1.875991 * Y) + (0.041556 * Z))
        b = ((0.055648 * X) + (-0.204043 * Y) + (1.057311 * Z))

        #Converting from linear RGB to non-linear RGB
        if(r < 0.00304) :
            r = 12.92 * r
        else :
            r = (1.055 * pow(r, (1 / 2.4))) - 0.055
            if ( r > 1 ) :
                r = 1
        if (g < 0.00304) :
            g = 12.92 * g
        else :
            g = (1.055 * pow(g, (1 / 2.4))) - 0.055
            if (g > 1 ) :
                g = 1
        if (b < 0.00304):
            b = 12.92 * b
        else :
            b = (1.055 * pow(b, (1 / 2.4))) - 0.055
            if (b > 1) :
                b = 1

        if math.isnan(r):
            r = 1
        if math.isnan(g):
            g = 1
        if math.isnan(b):
            b = 1

        #Converting from non-linear RGB to sRGB
        r = int(r * 255 + 0.5)
        g = int(g * 255 + 0.5)
        b = int(b * 255 + 0.5)

        inputImage[i,j] = b,g,r

outputImage = np.zeros([rows,cols,bands], dtype=np.uint8)

for i in range(0,rows):
    for j in range(0,cols):
        b, g, r = inputImage[i,j]
        outputImage[i,j] = [b,g,r]

cv2.imshow("output:",outputImage)
cv2.imwrite(outImage_name,outputImage)

cv2.waitKey(0)
cv2.destroyAllWindows()