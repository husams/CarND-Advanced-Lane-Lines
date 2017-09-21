import cv2
import numpy as np

# Lab (L = 0-100, A,B = -128, 128)
# Yuv (Y = 0-1, U,V = -0.5, 0.5)
# HLS (H = 0-360 , S,L = 0 - 100)
# HSV (H = 0-360, S,V = 0 - 100)

def abs_sobel_thresh(img, orient='x',sobel_kernel=3, thresh=(0,255), gray=None):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if gray is None else gray
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    abs_sobel = np.absolute(sobel)
    # 5) Create a mask of 1's where the scaled gradient magnitude
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255), gray=None):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if gray is None else gray
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Calculate the magnitude 
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2), gray=None):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if gray is None else gray
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction =  np.arctan2(abs_sobely , abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def color_threshold(image, color=cv2.COLOR_RGB2HLS, channel=2, thresh=(0, 255)):
    channels = image if color is None else cv2.cvtColor(image, color)
    img      = channels[:,:,channel]
    sbinary = np.zeros_like(img)
    sbinary[(img > thresh[0]) & (img <= thresh[1])] = 1
    return sbinary

def color_mask(image, lower_yellow, upper_yellow, lower_white, upper_white):
    yellow_mask = cv2.inRange(image, lower_yellow, upper_yellow)
    white_mask  = cv2.inRange(image, lower_white,  upper_white)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    return mask
