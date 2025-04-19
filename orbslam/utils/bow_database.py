import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
from scipy.spatial.distance import cdist
import pickle

class BoWDatabase:
    def __init__(self, k=10, depth=3):
        """
        Implements a Bag of Words (BoW) database for place recognition in ORB-SLAM.
        
        :param k: Number of clusters per tree level (default: 10)
        :param depth: Depth of hierarchical vocabulary tree (default: 3)
        """
        self.k = k
        self.depth = depth
        self.words = None
        self.word_weights = None
        self.bow_vectors = {}  # Stores BoW representations for images
        self.save_path = "/home/nitin/NitinWs/CustomOrbSlam/data/vocabulary.pkl"
    
    def train_vocabulary(self, descriptors):
        """
        Train vocabulary using MiniBatchKMeans clustering.
        
        :param descriptors: List of descriptors from multiple images.
        """
        descriptors = np.vstack(descriptors)  # Flatten all descriptors
        kmeans = MiniBatchKMeans(n_clusters=self.k**self.depth, batch_size=1000, random_state=0).fit(descriptors)
        self.words = kmeans.cluster_centers_
        
        # Compute IDF weights
        word_counts = np.zeros(len(self.words))
        for d in descriptors:
            unique_words = np.unique(self.get_visual_word(d))
            for w in unique_words:
                word_counts[w] += 1
        
        num_documents = len(descriptors)
        self.word_weights = np.log(num_documents / (1 + word_counts))
        
        # Save vocabulary
        with open(self.save_path, "wb") as f:
            pickle.dump((self.words, self.word_weights), f)
    
    def load_vocabulary(self):
        """
        Load the pre-trained vocabulary from file.
        """
        with open(self.save_path, "rb") as f:
            self.words, self.word_weights = pickle.load(f)
    
    def get_visual_word(self, descriptor):
        """
        Convert a descriptor to its nearest visual word in the vocabulary.
        
        :param descriptor: ORB descriptor to be quantized.
        :return: Index of the closest visual word.
        """
        if descriptor is None or descriptor.size == 0:
            return np.array([])
        descriptor = np.atleast_2d(descriptor)  # Ensure descriptor is 2D
        distances = cdist(descriptor, self.words, metric='euclidean')
        return np.argmin(distances, axis=1)
    
    def add_image(self, image_id, words):
        """
        Add an image's BoW representation to the database.
        
        :param image_id: Unique identifier for the image.
        :param words: List of visual words extracted from the image.
        """
        word_histogram = defaultdict(int)
        for w in words:
            word_histogram[w] += 1
        
        # Apply TF-IDF weighting
        for w in word_histogram:
            word_histogram[w] *= self.word_weights[w]
        
        self.bow_vectors[image_id] = word_histogram
    
    def query(self, query_words, top_k=12):
        """
        Query the database to find the top matching images.
        
        :param query_words: List of visual words from the query image.
        :param top_k: Number of best matches to return (default: 12).
        :return: List of (image_id, score) sorted by similarity.
        """
        scores = []
        for image_id, bow_vec in self.bow_vectors.items():
            common_words = set(query_words) & set(bow_vec.keys())
            score = sum(min(query_words.count(w) * self.word_weights[w], bow_vec[w]) for w in common_words)
            scores.append((image_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
