import cv2
import numpy as np



def channel_threshold(channel, min=0, max=255):
    binary = np.zeros_like(channel).astype(np.uint8)
    binary[(channel >= min) & (channel <= max)] = 1
    return binary

def color_threshold(image, min, max):
    c1_mask = channel_threshold(image[:,:,0], min=min[0], max=max[0])
    c2_mask = channel_threshold(image[:,:,1], min=min[1], max=max[1])
    c3_mask = channel_threshold(image[:,:,2], min=min[2], max=max[2])
    binary = np.zeros_like(c1_mask)
    binary[(c1_mask == 1) & (c2_mask==1) & (c3_mask==1)] = 1
    return binary


def binary_mask(image):
    hsv   = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    luv   = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    
    yellow_mask = color_threshold(hsv, [0,80,200],[255,255,255])
    white_mask  = color_threshold(hsv, [0,0,218],[255,255,255])
    luv_mask    = color_threshold(luv, [215,95,140], [254,106,148])
    
    combined_binary = np.zeros_like(yellow_mask)
    combined_binary[(yellow_mask == 1) | (white_mask == 1) | (luv_mask==1)] = 1
    return combined_binary


def get_perspective_transform(image, src, dst):
    image_size    = (image.shape[1], image.shape[0])
  
    M    = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

def warp(binary, M):
    image_size    = (binary.shape[1], binary.shape[0])
    return cv2.warpPerspective(binary, M, image_size, flags=cv2.INTER_LINEAR)