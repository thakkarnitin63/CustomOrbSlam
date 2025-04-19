import cv2
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
from scipy.spatial.distance import cdist
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

class ORBExtractor:
    def __init__(self):
        self.orb = cv2.ORB_create()
    
    def extract(self, image):
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors

class Vocabulary:
    def __init__(self, k=10, depth=3, weighting='TF_IDF', scoring='L1_NORM'):
        self.k = k
        self.depth = depth
        self.weighting = weighting
        self.scoring = scoring
        self.words = None
        self.word_weights = None
        self.save_path = "/home/nitin/NitinWs/CustomOrbSlam/data/vocabulary.pkl"
    
    def train(self, descriptors):
        print("Training vocabulary with MiniBatchKMeans...")
        descriptors = np.vstack(descriptors)  # Flatten all descriptors
        kmeans = MiniBatchKMeans(n_clusters=self.k**self.depth, batch_size=1000, random_state=0).fit(descriptors)
        self.words = kmeans.cluster_centers_
        print("Vocabulary training completed!")
        
        # Compute IDF weights
        print("Computing IDF weights...")
        word_counts = np.zeros(len(self.words))
        for d in tqdm(descriptors, desc="Calculating word frequencies"):
            unique_words = np.unique(self.get_visual_word(d))
            for w in unique_words:
                word_counts[w] += 1
        
        num_documents = len(descriptors)
        self.word_weights = np.log(num_documents / (1 + word_counts))
        print("IDF weights computation completed!")
        
        # Save vocabulary and weights
        print(f"Saving vocabulary to {self.save_path}")
        with open(self.save_path, "wb") as f:
            pickle.dump((self.words, self.word_weights), f)
        print("Vocabulary saved successfully!")
    
    def load(self):
        print(f"Loading vocabulary from {self.save_path}")
        with open(self.save_path, "rb") as f:
            self.words, self.word_weights = pickle.load(f)
        print("Vocabulary loaded successfully!")
    
    def get_visual_word(self, descriptor):
        if descriptor is None or descriptor.size == 0:
            return np.array([])
        descriptor = np.atleast_2d(descriptor)  # Ensure descriptor is always 2D
        distances = cdist(descriptor, self.words, metric='euclidean')
        return np.argmin(distances, axis=1)

class Database:
    def __init__(self, weighting='TF_IDF', scoring='L1_NORM'):
        self.bow_vectors = {}
        self.weighting = weighting
        self.scoring = scoring
        self.word_weights = None
    
    def set_word_weights(self, word_weights):
        self.word_weights = word_weights
    
    def add_image(self, image_id, words):
        word_histogram = defaultdict(int)
        for w in words:
            word_histogram[w] += 1
        
        # Apply TF-IDF weighting if needed
        for w in word_histogram:
            word_histogram[w] *= self.word_weights[w]
        
        self.bow_vectors[image_id] = word_histogram
    
    def query(self, query_words, top_k=12):
        scores = []
        
        # Compute similarity scores for all images
        for image_id, bow_vec in self.bow_vectors.items():
            common_words = set(query_words) & set(bow_vec.keys())
            score = sum(min(query_words.count(w) * self.word_weights[w], bow_vec[w]) for w in common_words)
            scores.append((image_id, score))

        # Sort by highest score and return the top_k matches
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]  # Return top 12 matches

# Load KITTI Dataset and Extract Features
def load_kitti_images(dataset_path):
    extractor = ORBExtractor()
    descriptors_list = []
    images = []
    image_filenames = sorted(os.listdir(dataset_path))
    
    print("Extracting ORB features from images...")
    for img_name in tqdm(image_filenames, desc="Processing images"):
        img_path = os.path.join(dataset_path, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = extractor.extract(image)
        if descriptors is not None:
            descriptors_list.append(descriptors)
            images.append(image)
    print("Feature extraction completed!")
    return images, descriptors_list

# Train Vocabulary and Database
def train_dbow2(kitti_path):
    images, descriptors = load_kitti_images(kitti_path)
    vocab = Vocabulary(k=10, depth=3)
    vocab.train(descriptors)
    
    # Initialize database
    db = Database()
    db.set_word_weights(vocab.word_weights)
    
    print("Adding images to database...")
    for i, d in tqdm(enumerate(descriptors), total=len(descriptors), desc="Indexing images"):
        words = vocab.get_visual_word(d)
        db.add_image(i, words)
    print("Database indexing completed!")
    
    return vocab, db

# Example usage
if __name__ == "__main__":
    KITTI_PATH = "/home/nitin/NitinWs/CustomOrbSlam/data/dataset/sequences/00/image_0"  # Update with actual path
    vocab, db = train_dbow2(KITTI_PATH)  # ✅ This only runs if we run this file directly!

