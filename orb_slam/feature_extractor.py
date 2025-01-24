import cv2

class FeatureExtractor:
    def __init__(self, nfeatures=500):
        """
        Initializes the FeatureExtractor with the specified number of features.
        :param nfeatures: The maximum number of features to retain.
        """
        self.nfeatures = nfeatures
        self.orb=cv2.ORB_create(nfeatures=self.nfeatures)

    def extract(self, image):
        """
        Detects keypoints and computes descriptors in the given image.
        :param image: Input grayscale image.
        :return: keypoints and descriptors.
        """
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors