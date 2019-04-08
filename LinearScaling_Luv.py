import numpy as np
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 11:30:14 2019
@author: Rishika Parashar 
"""
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

L_min = float('inf')
L_max = -float('inf')
Xw, Yw, Zw = [0.95, 1.0, 1.09]
uw = (4*Xw)/(Xw+(15*Yw)+(3*Zw))
vw = (9*Yw)/(Xw+(15*Yw)+(3*Zw))
A = 0
B = 100

# To get the minimum and maximum L values
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

        #Converting from XYZ to Luv domain
        t = Y/Yw
        if(t > 0.008856) :
            L = (116 * pow(t, 1/3)) - 16
        else :
            L = 903.3 * t

        if (L > L_max) :
            L_max = L
        if (L < L_min) :
            L_min = L
            
# linear scaling of L values and convert the image to back to sRGB.
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

        # Converting from XYZ to Luv domain
        t = Y / Yw
        if (t > 0.008856):
            L = (116 * pow(t, 1/3)) - 16
        else:
            L = 903.3 * t

        if (L_max == L_min) :
            L_scaled = L
        if( L < L_min) :
            L_scaled = 0
        elif( L > L_max) :
            L_scaled = 100
        else :
            #Linear scaling of L values in the range [A,B]
            L_scaled = (L - L_min) * ((B - A) / (L_max - L_min))

        #Calculating the u and v values after scaling the L values
        d = X + (15 * Y) + (3 * Z)
        if(d == 0):
            u1 = 4
            v1 = 9
        else :
            u1 = (4*X)/d
            v1 = (9*Y)/d
        u = (13*L_scaled)*(u1-uw)
        v = (13*L_scaled)*(v1-vw)

        #Converting back to XYZ domain from Luv domain
        if(L_scaled == 0) :
            u1 = uw
            v1 = vw
        else :
            u1 = (u + (13*uw*L_scaled))/(13*L_scaled)
            v1 = (v + (13*vw*L_scaled))/(13*L_scaled)

        if (L_scaled > 7.9996) :
            Y = pow(((L_scaled + 16)/116),3)*Yw
        else :
            Y = (L_scaled/903.3) * Yw

        if(v1 == 0) :
            X = 0
            Z = 0
        else :
            X = Y * (2.25 *(u1/v1))
            Z = Y * (3 - (0.75 * u1) - (5 * v1))/v1


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