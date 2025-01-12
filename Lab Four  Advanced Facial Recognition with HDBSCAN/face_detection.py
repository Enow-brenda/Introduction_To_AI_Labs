from skimage.feature import hog, local_binary_pattern
import cv2
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

'''
Step 1: Data Preparation
1. Dataset:
Use the lfw_dataset from scikit-learn, which contains images of faces.
Each image can be flattened into a 1D array or reshaped for processing.
2. Image Preprocessing:
Grayscale Conversion: Convert images to grayscale to reduce complexity.
Normalization: Scale pixel values to [0, 1] to standardize the data.
Resize Images: Ensure all images have the same dimensions (e.g., 64x64).
'''
# Load the LFW dataset
lfw_dataset = fetch_lfw_people(min_faces_per_person=50, resize=0.4)
images = lfw_dataset.images  # Shape: (n_samples, height, width)
labels = lfw_dataset.target

# Preprocessing

preprocess_images = []
for img in images:
    if len(img.shape) == 3:  # Check if the image has 3 channels
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    else:  # Image is already grayscale
        img_gray = img
    img_resized = cv2.resize(img_gray, (64, 64))  # Resize
    img_normalized = img_resized / 255.0  # Normalize
    preprocess_images.append(img_normalized)

preprocessed_images = np.array(preprocess_images)

'''

Step 2: Feature Extraction
Extract features from each image using handcrafted techniques (since you cannot use neural networks):
Edge Detection: Use algorithms like Sobel or Canny to extract edges (important for face detection).
Histogram of Oriented Gradients (HOG): A popular feature extraction technique for object detection.
Local Binary Patterns (LBP): Captures texture patterns in an image, helpful for face identification.'''
def extract_features(images):
    features = []
    for img in images:
        # Extract HOG features
        hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        # Extract LBP features
        lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
        lbp_hist = np.histogram(lbp.ravel(), bins=np.arange(0, 27), density=True)[0]
        # Combine features
        combined_features = np.hstack((hog_features, lbp_hist))
        features.append(combined_features)
    return np.array(features)

features = extract_features(preprocessed_images)
print("features: ",features)

# PCA for dimensionality reduction
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA(n_components=50)
reduced_features = pca.fit_transform(features_scaled)

# HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
labels = clusterer.fit_predict(reduced_features)

# Analyze clustering results
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Number of clusters: {num_clusters}")
print(f"Outliers (No Face): {np.sum(labels == -1)}")

