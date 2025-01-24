import numpy as np
import cv2

class PoseEstimator:
    def __init__(self, K):
        self.K = K

    def estimate(self, kp1, kp2, matches):
        #Extract matched keypoints
        pts1= np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        #Compute Essential Matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        #Recover pose from Essential Matrix
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)

        return R, t