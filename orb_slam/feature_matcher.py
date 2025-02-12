import cv2

class FeatureMatcher:
    def __init__(self, ratio_thresh=0.75, orentation_thresh=30):
        """
        Initializes the FeatureMatcher with the specified ratio threshold.
        :param ratio_thresh: The ratio threshold for filtering ambiguous matches.
        :param orentation_thresh: Threshold for filtering matches based on orentation consistency.
        """
        self.ratio_thresh = ratio_thresh
        self.orientation_thresh = orentation_thresh
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, des1, des2, kp1, kp2):
        """
        Matches descriptors between two images using the BFMatcher and applies the ratio test.
        :param des1: Descriptors from the first image.
        :param des2: Descriptors from the second image.
        :param kp1: Keypoints from first image.
        :param kp2: Keypoints from second image.
        :return: List of filtered good matches.
        """
        # Perform k-nearest neighbors matching
        knn_matches = self.bf.knnMatch(des1, des2, k=2)


        # Apply the ratio test to filter ambiguous matches
        good_matches = [
            m for m, n in knn_matches if m.distance < self.ratio_thresh * n.distance
        ]
        filtered_matches = []
        for match in good_matches:
            keyp1 = kp1[match.queryIdx]
            keyp2 = kp2[match.trainIdx]

            # Calculate the angle difference between keypoints
            angle_diff = abs(keyp1.angle - keyp2.angle)
            if angle_diff < self.orientation_thresh or abs(angle_diff - 360) < self.orientation_thresh:
                filtered_matches.append(match)

        return filtered_matches
    
    def visualize_matches(self, img1, img2, keypoints1, keypoints2, matches, title="Matches"):
        """
        Visualizes matches between two images.
        :param img1: The first image.
        :param img2: The second image.
        :param keypoints1: Keypoints from the first image.
        :param keypoints2: Keypoints from the second image.
        :param matches: List of matches to visualize.
        :param title: Title for the visualization window.
        """
        # Draw matches
        img_matches = cv2.drawMatches(
            img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imshow(title, img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

