import random
import numpy as np
from orb_slam.vocab_create import Vocabulary, Database, load_kitti_images

# ✅ Load the saved vocabulary instead of retraining
print("Loading saved vocabulary...")
vocab = Vocabulary()
vocab.load()

# ✅ Print vocabulary stats
print(f"Loaded vocabulary with {len(vocab.words)} words.")

# ✅ Initialize the database and load word weights
print("Reconstructing database...")
db = Database()
db.set_word_weights(vocab.word_weights)

# ✅ Load KITTI dataset without retraining
KITTI_PATH = "/home/nitin/NitinWs/CustomOrbSlam/data/dataset/sequences/00/image_0"
_, descriptors = load_kitti_images(KITTI_PATH)  # Only extract features

# ✅ Populate the database
for i, d in enumerate(descriptors):
    words = vocab.get_visual_word(d)
    db.add_image(i, words)

print(f"Database loaded with {len(db.bow_vectors)} images.")

# ✅ Select a random image for testing
random_index = random.randint(0, len(db.bow_vectors) - 1)
query_descriptors = descriptors[random_index]

# ✅ Convert descriptors to visual words
query_words = vocab.get_visual_word(query_descriptors)

# ✅ Convert NumPy array to Python list
query_words = query_words.tolist()

# ✅ Query the database for the top 12 best matches
top_matches = db.query(query_words, top_k=12)

print(f"Query image index: {random_index}")
print("Top 12 matching images and scores:")
for match, score in top_matches:
    print(f"- Image {match}, Score: {score}")
