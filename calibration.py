import numpy as np
import cv2
import glob
import matplotlib.image as im

class Camera(object):
    def __init__(self):
        self.mtx  = None
        self.dist = None

    def calibrate(self, imagesPath):
        """
            Calibrate camera using set of images
        Args:
            imagesPath(string)  - path to input images
        Returns:
            None
        """
        # Object points
        objp        = np.zeros((6*9, 3), np.float32)
        objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

        objpoints = []
        imgpoints = []

        images = glob.glob(imagesPath)

        for idx, filename in enumerate(images):
            # Read image
            image = im.imread(filename)
            # Convert to gray scale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Find image points
            isFound, points = cv2.findChessboardCorners(gray, (9,6), None)
            
            # Check if we found an image
            if isFound == True:
                objpoints.append(objp)
                imgpoints.append(points)

        cv2.destroyAllWindows()

        # Read test image
        image = im.imread(images[0])
        size  = (image.shape[1], image.shape[0])

        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)

        self.mtx  = mtx
        self.dist = dist
    
    def  undistort(self, image):
        """
            Undistort image
        Args:
            image(np.array)  - (width x height x channel) distorted image. 
        Returns:
            undistort image
        """
        if self.mtx is None or self.dist is None:
            raise Exception("You need to calibrate the camera first")

        return cv2.undistort(image, self.mtx,self.dist, None, self.mtx)