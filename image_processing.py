import cv2
import numpy as np



def channel_threshold(channel, min=0, max=255):
    """
         convert single channel to binray image using 
         a given range.
        Args:
            channel(np.array) - (heigt x width) matrix represnt single channel.
            min(int)          - Thrshold lower bound
            max(int)          - Thrshold upper bound
        Returns:
            A (height x Width) array which respreset binray mask 
            for single channel
    """ 

    binary = np.zeros_like(channel).astype(np.uint8) 
    binary[(channel >= min) & (channel <= max)] = 1
    return binary

def color_threshold(image, min, max):
    """
         convert image to binray image using 
         a given range.
        Args:
            image(np.array)   - (heigt x width x channels) matrix represnt an colored image.
            min(int)          - Thrshold lower bound
            max(int)          - Thrshold upper bound
        Returns:
            A (height x Width x 1) array which respreset binray mask 
            for an image
    """ 
    c1_mask = channel_threshold(image[:,:,0], min=min[0], max=max[0])
    c2_mask = channel_threshold(image[:,:,1], min=min[1], max=max[1])
    c3_mask = channel_threshold(image[:,:,2], min=min[2], max=max[2])
    binary = np.zeros_like(c1_mask)
    binary[(c1_mask == 1) & (c2_mask==1) & (c3_mask==1)] = 1
    return binary


def binary_mask(image):
    """
        Use color threash hold using HSV/LUV color space to
        create mask for the white and yellow lines.
        Args:
            image(np.array) - (heigt x width x channels) matrix represnt an colored image.
        Returns:
            A (height x Width x 1) array which respreset binray image with 
            white and yellow lines marked.
    """ 
    hsv   = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    luv   = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    
    yellow_mask = color_threshold(hsv, [0,80,200],[255,255,255])
    white_mask  = color_threshold(hsv, [0,0,218],[255,255,255])
    luv_mask    = color_threshold(luv, [215,95,140], [254,106,148])
    
    combined_binary = np.zeros_like(yellow_mask)
    combined_binary[(yellow_mask == 1) | (white_mask == 1) | (luv_mask==1)] = 1
    return combined_binary


def get_perspective_transform(image, src, dst):
    """
        Conpute the perspective transform matrix
        using source and destination points
        Args:
            image(np.array)   - (heigt x width x channels) matrix represnt an colored image.
            src(np.array)     - array hold list of points  from source image
            dst(int)          - array hold list of points  from destination image (warped)
        Returns:
            Perspective transform matrix and the inverse
    """ 
    image_size    = (image.shape[1], image.shape[0])
  
    M    = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

def warp(binary, M):
    """
        Transform an image to brid view.
        Args:
            image(np.array)   - (heigt x width x channels) matrix represnt an colored image.
            M(np.array)       - Transform matrix.
        Returns:
             Transformed image.
    """ 
    image_size    = (binary.shape[1], binary.shape[0])
    return cv2.warpPerspective(binary, M, image_size, flags=cv2.INTER_LINEAR)