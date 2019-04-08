# Image-processing---Linear-Scaling
Applying Linear scaling on an image using opencv and python. 
Program is related to the manipulation of color in digital images.
The program changes the color of the image based on a histogram computed from a window in the
image. The window is specified in terms of the normalized coordinates w1, h1, w2, h2, where the window
upper left point is (w1; h1), and its lower right point is (w2; h2). For example, w1 = 0, h1 = 0, w2 = 1,
h2 = 1 is the entire image, and w1 = 0:3, h1 = 0:3, w2 = 0:7, h2 = 0:7 is is window in the center of the
image.
This program gets as input a color image, performs linear scaling in the Luv domain, and writes the
scaled image as output.
Pixel values outside the window are changed. Only pixels within the window are changed.
The scaling in Luv should stretch only the luminance values and map the smallest L value in the specied window to 0, and the largest L value in the specified window to 100.
