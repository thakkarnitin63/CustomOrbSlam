import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self, nfeatures=2000, scaleFactor=1.2, nlevels=8, iniThFAST=20, minThFAST=7):
        """
        Initializes the FeatureExtractor with the specified parameters.
        :param nfeatures: The maximum number of features to retain.
        :param scaleFactor: Pyramid decimation ratio.
        :param nlevels: The number of levels in the pyramid.
        :param iniThFAST: Initial FAST threshold for corner detection.
        :param minThFAST: Minimum FAST threshold for low-contrast areas.
        """
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scaleFactor,
            nlevels=nlevels,
            edgeThreshold=31,  # Standard edge size used in ORB-SLAM
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=iniThFAST
        )
        self.minThFAST = minThFAST

    def extract(self, image):
        """
        Detects keypoints and computes descriptors in the given image.
        Applies dynamic adjustment for low-contrast areas.
        :param image: Input grayscale image.
        :return: keypoints and descriptors.
        """
        # Detect keypoints using the initial FAST threshold
        keypoints = self.orb.detect(image, None)

        # Dynamically lower the threshold if not enough keypoints are detected
        if len(keypoints) < self.orb.getMaxFeatures():
            self.orb.setFastThreshold(self.minThFAST)
            keypoints = self.orb.detect(image, None)
            # Reset to the initial threshold
            self.orb.setFastThreshold(self.orb.getFastThreshold())

        # Compute descriptors with the final set of keypoints
        keypoints, descriptors = self.orb.compute(image, keypoints)
        return keypoints, descriptors
