import cv2

class FeatureExtractor:
    def __init__(self, nfeatures=500):
        self.orb=cv2.ORB_create(nfeatures=nfeatures)

    def extract(self, image):
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors