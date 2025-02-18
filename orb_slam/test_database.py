import os
import cv2
import csv
import tarfile
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Import your FeatureExtractor and vocabulary classes.
from orb_slam.feature_extractor import FeatureExtractor
from orb_slam.vocab_create import ORBVocabulary, VocabularyNode

# -------------------------------------------------------
# Function to load ORB vocabulary from a text file inside a tar.gz archive.
# -------------------------------------------------------
def load_orb_vocabulary(tar_filename, max_words=1000):
    """
    Loads an ORB vocabulary from a compressed tar.gz file containing ORBvoc.txt.
    This parser converts each nonempty line into floats and, if there are
    at least 32 numbers, takes the last 32 values as the descriptor.
    Then, it constructs a dummy vocabulary with a single root node whose children
    are the descriptors.
    
    To avoid an excessively large vocabulary, the descriptors are downsampled
    to 'max_words' if necessary.
    """
    with tarfile.open(tar_filename, "r:gz") as tar:
        # Look for the member that contains "ORBvoc.txt" in its name.
        voc_member = None
        for member in tar.getmembers():
            if "ORBvoc.txt" in member.name:
                voc_member = member
                break
        if voc_member is None:
            raise Exception("ORBvoc.txt not found in the archive.")
        
        f = tar.extractfile(voc_member)
        if f is None:
            raise Exception("Failed to extract ORBvoc.txt from archive.")
        content = f.read().decode("utf-8")
    
    lines = content.splitlines()
    descriptor_list = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines.
        tokens = line.split()
        try:
            # Attempt to convert all tokens to float.
            numbers = [float(tok) for tok in tokens]
        except ValueError:
            # Skip lines that can't be converted (likely header lines).
            continue
        # If there are at least 32 numbers, assume the last 32 are the descriptor.
        if len(numbers) >= 32:
            descriptor = np.array(numbers[-32:], dtype=np.float32)
            descriptor_list.append(descriptor)
    
    print(f"Loaded {len(descriptor_list)} descriptors from the vocabulary file.")
    
    # Downsample the vocabulary if too many descriptors are loaded.
    if len(descriptor_list) > max_words:
        descriptor_list = descriptor_list[:max_words]
        print(f"Downsampled vocabulary to {len(descriptor_list)} descriptors.")
    
    # Create a dummy vocabulary:
    # Create a dummy root node that has each descriptor as a child node.
    root = VocabularyNode(center=None, word_id=None)
    root.children = {i: VocabularyNode(center=desc, word_id=i) for i, desc in enumerate(descriptor_list)}
    
    vocab = ORBVocabulary()
    vocab.vocab_tree = root
    # Assign dummy IDF weights (e.g., 1.0 for every word)
    vocab.word_idf = {i: 1.0 for i in range(len(descriptor_list))}
    
    return vocab

# -------------------------------------------------------
# Helper: Compute similarity between two BoW vectors.
# -------------------------------------------------------
def compute_similarity(bow1, bow2):
    """
    Compute similarity as the dot product of the two BoW vectors.
    Both vectors are assumed to be normalized.
    """
    score = 0.0
    for word, weight in bow1.items():
        if word in bow2:
            score += weight * bow2[word]
    return score

# -------------------------------------------------------
# Feature Extraction Function
# -------------------------------------------------------
def extract_features(image_paths):
    """
    Extracts ORB features from a list of image paths using the FeatureExtractor.
    Returns a list of descriptor arrays.
    """
    extractor = FeatureExtractor()
    feature_list = []

    for img_path in tqdm(image_paths, desc="Extracting ORB Features"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Could not read {img_path}")
            continue

        keypoints, descriptors = extractor.extract(img)
        if descriptors is not None:
            feature_list.append(descriptors)
        else:
            feature_list.append(np.array([]))
    return feature_list

# -------------------------------------------------------
# Test Loop Closure / Place Recognition on Full Dataset
# -------------------------------------------------------
def test_loop_closure(vocab, image_paths, top_n=5, output_csv="loop_closure_results.csv"):
    """
    Transforms all images into BoW vectors and then compares each image
    against all others to find the top matching images.
    The top matches for each image are saved in a CSV file.
    """
    print("Extracting features for loop closure test...")
    features = extract_features(image_paths)

    print("Computing BoW vectors for each image...")
    bow_vectors = []
    for desc in tqdm(features, desc="Transforming to BoW"):
        if desc.size == 0:
            bow_vectors.append({})
        else:
            bow_vectors.append(vocab.transform(desc))

    num_images = len(bow_vectors)
    similarity_matrix = np.zeros((num_images, num_images), dtype=np.float32)

    print("Computing similarity matrix...")
    for i in tqdm(range(num_images), desc="Comparing images"):
        for j in range(i, num_images):
            score = compute_similarity(bow_vectors[i], bow_vectors[j])
            similarity_matrix[i, j] = score
            similarity_matrix[j, i] = score  # symmetric

    # Prepare CSV rows.
    csv_rows = []
    csv_rows.append(["Image", "Rank", "Matched Image", "Similarity"])

    print("\nTop matches for each image:")
    for i in range(num_images):
        scores = similarity_matrix[i].copy()
        scores[i] = -1  # Exclude self-match.
        top_indices = np.argsort(-scores)[:top_n]
        top_scores = scores[top_indices]
        print(f"Image {i}:")
        for rank, (idx, s) in enumerate(zip(top_indices, top_scores), start=1):
            print(f"   Rank {rank}: Image {idx} with similarity {s:.4f}")
            csv_rows.append([f"Image {i}", rank, f"Image {idx}", f"{s:.4f}"])

    # Save the results to CSV.
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)

    print(f"\n✅ Loop closure results saved to {output_csv}")

# -------------------------------------------------------
# Main testing function
# -------------------------------------------------------
def main():
    # Path to your KITTI image sequence.
    IMAGE_DIR = "/home/nitin/NitinWs/CustomOrbSlam/data/dataset/sequences/00/image_0"
    num_images = 4541
    image_paths = [os.path.join(IMAGE_DIR, f"{i:06d}.png") for i in range(num_images)]

    # Load your ORB vocabulary from the ORBvoc.txt.tar.gz file.
    vocab_file = "/home/nitin/NitinWs/CustomOrbSlam/data/ORBvoc.txt.tar.gz"
    vocab = load_orb_vocabulary(vocab_file, max_words=1000)

    # Run the loop closure test and save results to CSV.
    test_loop_closure(vocab, image_paths, top_n=5, output_csv="loop_closure_results.csv")

if __name__ == "__main__":
    main()
