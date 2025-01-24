import cv2

class FeatureMatcher:
    def __init__(self, ratio_thresh=0.75):
        """
        Initializes the FeatureMatcher with the specified ratio threshold.
        :param ratio_thresh: The ratio threshold for filtering matches.
        """
        self.ratio_thresh = ratio_thresh
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, des1, des2):
        """
        Matches descriptors between two images using the BFMatcher and applies the ratio test.
        :param des1: Descriptors from the first image.
        :param des2: Descriptors from the second image.
        :return: List of good matches.
        """
        # Perform k-nearest neighbors matching
        knn_matches = self.bf.knnMatch(des1, des2, k=2)

        # Apply the ratio test to filter matches
        good_matches = []
        for m, n in knn_matches:
            if m.distance < self.ratio_thresh * n.distance:
                good_matches.append(m)

        return good_matches
