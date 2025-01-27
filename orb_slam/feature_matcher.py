import cv2

class FeatureMatcher:
    def __init__(self, ratio_thresh=0.75):
        """
        Initializes the FeatureMatcher with the specified ratio threshold.
        :param ratio_thresh: The ratio threshold for filtering ambiguous matches.
        """
        self.ratio_thresh = ratio_thresh
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, des1, des2):
        """
        Matches descriptors between two images using the BFMatcher and applies the ratio test.
        :param des1: Descriptors from the first image.
        :param des2: Descriptors from the second image.
        :return: List of filtered good matches.
        """
        # Perform k-nearest neighbors matching
        knn_matches = self.bf.knnMatch(des1, des2, k=2)

        # Apply the ratio test to filter matches
        good_matches = []
        for m, n in knn_matches:
            if m.distance < self.ratio_thresh * n.distance:
                good_matches.append(m)

        return good_matches

    def filter_matches(self, matches, keypoints1, keypoints2, threshold=30):
        """
        Filters matches based on geometric constraints like orientation consistency.
        :param matches: List of matches to filter.
        :param keypoints1: Keypoints from the first image.
        :param keypoints2: Keypoints from the second image.
        :param threshold: Angular threshold for orientation consistency in degrees.
        :return: Filtered matches.
        """
        filtered_matches = []
        for match in matches:
            kp1 = keypoints1[match.queryIdx]
            kp2 = keypoints2[match.trainIdx]

            # Calculate the angle difference
            angle_diff = abs(kp1.angle - kp2.angle)
            if angle_diff < threshold or abs(angle_diff - 360) < threshold:
                filtered_matches.append(match)

        return filtered_matches
