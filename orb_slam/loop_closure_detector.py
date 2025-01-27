import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class LoopClosureDetector:
    def __init__(self, feature_extractor, num_clusters=500):
        """
        Initializes the LoopClosureDetector.

        Parameters:
        - feature_extractor: An instance of FeatureExtractor for extracting features from images.
        - num_clusters: Number of clusters for the visual vocabulary (default: 500).

        """
        self.feature_extractor = feature_extractor
        self.num_clusters = num_clusters
        self.vocabulary = None
        self.keyframe_descriptors = []
        self.keyframe_bow_histrograms = []


    def create_vocabulary(self, images):

        """
        Creates a visual vocabulary from a list of images.

        Parameters:
        - images: List of images to build the vocabulary.
        """

        all_descriptors=[]

        # Extract descriptors from each image

        for image in images:
            _, descriptors = self.feature_extractor.extract(image)
            if descriptors is not None:
                all_descriptors.append(descriptors)


        # Stack all descriptors vertically
        all_descriptors = np.vstack(all_descriptors)

        # Perform K-Means Clustering to create the vocabulary
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
        kmeans.fit(all_descriptors)
        self.vocabulary = kmeans.cluster_centers_ 

    
    def compute_bow_histogram(self, descriptors):
        """
        Computes the Bag of Words histogram for a set of descriptors.

        Parameters:
        - descriptors: Feature descriptors of an image.

        Returns:
        - bow_histogram: Normalized histogram representing the BoW model.
        """

        if self.vocabulary is None:
            raise ValueError("Vocabulary has not been created.")
        
        # Compute distances between descriptors and vocabulary words
        distances = cdist(descriptors, self.vocabulary, metric='euclidean')

        # Find the nearest vocabulary word for each descriptor
        closest_words = np.argmin(distances, axis=1)

        # Create histogram of visual words
        bow_histogram, _ = np.histogram(closest_words, bins=np.arange(self.num_clusters+1),density=True)


        return bow_histogram
    
    def add_keyframe(self, image):
        """
        Adds a keyframe to the dataset.

        Parameters:
        - image: 
        """
        keypoints, descriptors = self.feature_extractor.extract(image)
        if descriptors is not None:
            bow_histogram = self.compute_bow_histogram(descriptors)
            self.keyframe_descriptors.append(descriptors)
            self.keyframe_bow_histrograms.append(bow_histogram)

    def detect_loop_closure(self, image, similarity_threshold=0.8):
        """
        Detects loop closure by comparing the current image to stored keyframes.

        Parameters:
        - image: The current image to check for loop closure 
        - similarity_threshold : Threshold for considering a match (default:0.8)

        Returns:
        - loop_index: Index of matching keyframe if a loop is detected; otherwise, None.
        """       

        keypoints, descriptors = self.feature_extractor.extract(image)
        if descriptors is None:
            return None
        
        current_bow_histogram = self.compute_bow_histogram(descriptors)

        # Compare with existing keyframe histograms
        for index, keyframe_histogram in enumerate(self.keyframe_bow_histrograms):
            similarity = np.dot(current_bow_histogram, keyframe_histogram)
            if similarity > similarity_threshold:
                return index
            
        return None
    
    def geometric_verification(self, image, loop_index, ransac_thresh=5.0):
        """
        Performs geometric verification to confirm loop closure.

        Parameters:
        - image: The current image.
        - loop_index: Index of the candidate keyframe.
        -ransac_thresh: Ransac reprojection threshold (default: 5.0).

        Returns:
        -is_valid: Boolen indicating if the loop closure is valid.
        -transformation: The estimated transformation matrix if valid; otherwise, None.

        """
        keypoints1, descriptors1 =  self.feature_extractor.extract(image)
        keypoints2, descriptors2 = self.feature_extractor.extract(self.keyframe_descriptors[loop_index])


        # Match features between current image and candidate keyframe
        bf =cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches =bf.match(descriptors1, descriptors2)

        if len(matches)< 4:
            return False, None
        

        #Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # Estimate transformation using RANSAC
        transformation, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

        if transformation is not None:
            return True, transformation
        else:
            return False, None       


