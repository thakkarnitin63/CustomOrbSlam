import os
import cv2
import numpy as np
import pickle
from sklearn.cluster import KMeans
from collections import defaultdict
from tqdm import tqdm  # Progress bar for feature extraction

# Import FeatureExtractor
from orb_slam.feature_extractor import FeatureExtractor

# ------------------------------------------------
# DBoW2-style ORB Vocabulary Implementation
# ------------------------------------------------
class ORBVocabulary:
    def __init__(self, k=10, L=5, weighting="TF-IDF", scoring="L1-NORM"):
        """
        Implements a hierarchical k-means based Bag-of-Words (BoW) vocabulary.

        Parameters:
          - k: Branching factor (number of clusters at each level)
          - L: Number of levels in the vocabulary tree
          - weighting: "TF-IDF" for term frequency-inverse document frequency
          - scoring: "L1-NORM" for normalized similarity measurement
        """
        self.k = k
        self.L = L
        self.weighting = weighting
        self.scoring = scoring
        self.vocab_tree = None
        self.word_idf = None  # IDF weights for vocabulary words

    def _hierarchical_kmeans(self, descriptors, level=0):
        """Recursively applies k-means clustering to build the hierarchical tree."""
        if level >= self.L or len(descriptors) < self.k:
            return np.mean(descriptors, axis=0) if len(descriptors) > 0 else None

        kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        kmeans.fit(descriptors)

        clusters = defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(descriptors[i])

        return {label: self._hierarchical_kmeans(np.array(clusters[label]), level + 1)
                for label in clusters}

    def create(self, feature_list):
        """
        Trains the vocabulary tree using hierarchical k-means clustering.

        Parameters:
          - feature_list: List of descriptor arrays from multiple images.
        """
        all_descriptors = np.vstack(feature_list)  # Stack all descriptors into one array
        print(f"📌 Training vocabulary with {len(all_descriptors)} total descriptors...")

        self.vocab_tree = self._hierarchical_kmeans(all_descriptors)

        # Compute IDF weights for words (for TF-IDF weighting)
        num_images = len(feature_list)
        word_occurrences = np.zeros(len(all_descriptors))
        for features in feature_list:
            unique_words = set(self.transform(features))
            for word in unique_words:
                word_occurrences[word] += 1

        self.word_idf = np.log((num_images + 1) / (word_occurrences + 1))

    def transform(self, descriptors):
        """
        Converts descriptors into a Bag-of-Words (BoW) vector using the trained vocabulary.

        Parameters:
          - descriptors: ORB descriptors from an image.

        Returns:
          - bow_vector: Dictionary of word IDs and their weighted frequencies.
        """
        bow_vector = defaultdict(float)
        for desc in descriptors:
            word_id = self._find_nearest_word(desc, self.vocab_tree)
            bow_vector[word_id] += 1.0

        # Apply TF-IDF weighting if enabled
        if self.weighting == "TF-IDF":
            for word_id in bow_vector.keys():
                bow_vector[word_id] *= self.word_idf[word_id]

        # Normalize using L1 norm if required
        if self.scoring == "L1-NORM":
            norm = sum(bow_vector.values())
            if norm > 0:
                for word_id in bow_vector.keys():
                    bow_vector[word_id] /= norm

        return bow_vector

    def _find_nearest_word(self, descriptor, node, level=0):
        """Finds the closest cluster center in the vocabulary tree for a descriptor."""
        if not isinstance(node, dict):  # If it's a leaf node
            return node

        min_distance = float('inf')
        best_label = None

        for label, cluster_center in node.items():
            dist = np.linalg.norm(descriptor - cluster_center)
            if dist < min_distance:
                min_distance = dist
                best_label = label

        return self._find_nearest_word(descriptor, node[best_label], level + 1)

    def save(self, filename):
        """Saves the vocabulary as a pickle file."""
        with open(filename, "wb") as f:
            pickle.dump((self.k, self.L, self.weighting, self.scoring, self.vocab_tree, self.word_idf), f)
        print(f"✅ Vocabulary saved as {filename}")

    def load(self, filename):
        """Loads the vocabulary from a pickle file."""
        with open(filename, "rb") as f:
            self.k, self.L, self.weighting, self.scoring, self.vocab_tree, self.word_idf = pickle.load(f)
        print(f"✅ Vocabulary loaded from {filename}")


# -------------------------------------------------------
# FEATURE EXTRACTION FROM KITTI DATASET
# -------------------------------------------------------
def extract_features(image_paths):
    """
    Extracts ORB features from a list of image paths using the FeatureExtractor.

    Parameters:
      - image_paths: List of paths to input images.

    Returns:
      - feature_list: List of descriptor arrays (one per image).
    """
    extractor = FeatureExtractor()  # Using the given FeatureExtractor class
    feature_list = []

    for img_path in tqdm(image_paths, desc="🔍 Extracting Features"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Could not read {img_path}")
            continue

        keypoints, descriptors = extractor.extract(img)
        if descriptors is not None:
            feature_list.append(descriptors)

    return feature_list

# Path to KITTI dataset
IMAGE_DIR = "/home/nitin/NitinWs/CustomOrbSlam/data/dataset/sequences/00/image_0"

def extract_features():
    """
    Extracts ORB features from all KITTI images in the given directory.
    
    Returns:
      - feature_list: List of descriptor arrays (one per image).
    """
    extractor = FeatureExtractor()  # Using your FeatureExtractor class
    feature_list = []
    
    # Generate paths for images from 000000.png to 004540.png
    image_paths = [os.path.join(IMAGE_DIR, f"{i:06d}.png") for i in range(4541)]

    for img_path in tqdm(image_paths, desc="🔍 Extracting ORB Features"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Could not read {img_path}")
            continue

        keypoints, descriptors = extractor.extract(img)
        if descriptors is not None:
            feature_list.append(descriptors)

    return feature_list


def train_vocabulary():
    """
    Trains the visual vocabulary on KITTI images and saves it using Pickle.
    """
    # Extract ORB descriptors
    features = extract_features()

    # Create Vocabulary
    voc = ORBVocabulary(k=10, L=5, weighting="TF-IDF", scoring="L1-NORM")
    voc.create(features)
    
    # Save vocabulary using Pickle
    vocab_path = "/home/nitin/NitinWs/CustomOrbSlam/data/vocab/orb_vocabulary.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(voc, f)

    print(f"✅ KITTI Vocabulary created and saved at: {vocab_path}")


# -------------------------------------------------------
if __name__ == "__main__":
    train_vocabulary()