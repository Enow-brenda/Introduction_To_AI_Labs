import joblib
import cv2
import numpy as np
import hdbscan
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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


# Converting the image list into numpy array
X = np.array(images)
y = np.array(labels)

print("Number of images: ", X.size)
print("Labels: ", labels[:5])

# Displaying first 5 images
fig, ax = plt.subplots(1, 5, figsize=(15, 10))
for i, image in enumerate(images[:5]):
    ax[i].imshow(image, cmap='gray')  # Show grayscale images
    ax[i].set_title(labels[i])
    ax[i].axis('off')
plt.show()


def resize_image(image, target_size=(250, 250)):
    # Convert the NumPy array to a PIL image
    pil_image = Image.fromarray(image)

    # Resize the image using PIL's resize method
    resized_image = pil_image.resize(target_size)

    # Convert back to a NumPy array and return
    return np.array(resized_image)


# Resize the images
resized_images = [resize_image(img) for img in images]

# Flatten the resized images
flattened_images = [img.flatten() for img in resized_images]
X = np.array(flattened_images)



# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Instantiate PCA and fit it to the data
pca = PCA()
pca.fit(X_scaled)

# Check the explained variance ratio for each principal component or features
print("PCA explained variance ratio: ", pca.explained_variance_ratio_)

# 1. Using variance expectation
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Components')
plt.grid()
plt.legend()
plt.show()

# Print optimal number of components for 90% and 95% variance explained
optimal_90 = np.argmax(cumulative_variance >= 0.90) + 1
optimal_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"Optimal number of components for 90% variance: {optimal_90}")
print(f"Optimal number of components for 95% variance: {optimal_95}")

# Using the number of components for 90% variance
pca2 = PCA(n_components=optimal_90)
X_reduced = pca2.fit_transform(X_scaled)  # Make sure to use transform here

# Configuration for HDBSCAN
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=3,       # Minimum size of clusters
    min_samples=2,# Minimum samples in the neighborhood of a point
    metric='euclidean'
)

# Apply HDBSCAN
cluster_labels = clusterer.fit_predict(X_reduced)


# Extracting meaningful clusters
unique_labels = set(cluster_labels)
n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))

# Display results
print(f"Number of clusters detected: {n_clusters}")
print(f"Cluster labels: {cluster_labels}")

# 1. Simulating Noisy Data (Add Gaussian Noise)
def add_noise(data, noise_factor=0.2):
    noisy_data = data + noise_factor * np.random.randn(*data.shape)
    noisy_data = np.clip(noisy_data, 0, 255)  # Ensure the pixel values stay within valid range
    return noisy_data

# Simulate noisy data
X_noisy = add_noise(X_scaled)
X_reduced_noisy = pca2.transform(X_noisy)  # Noisy data


# Function to perform clustering and visualize results
def cluster_and_visualize(X_reduced, distance_metric='euclidean'):
    # Configuration for HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,  # Minimum size of clusters
        min_samples=2,  # Minimum samples in the neighborhood of a point
        metric=distance_metric , # Distance metric
        cluster_selection_method = 'leaf'
    )

    # Apply HDBSCAN
    cluster_labels = clusterer.fit_predict(X_reduced)

    # Visualize the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.5)
    plt.title(f'HDBSCAN Clustering Results ({distance_metric} distance)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster Label')
    plt.show()

    return cluster_labels


# 2. Clustering on Clean Data (Euclidean Distance)
print("Clustering on Clean Data (Euclidean Distance)")
cluster_labels_clean = cluster_and_visualize(X_reduced, distance_metric='euclidean')
euclidean_clean_silhoute_score  = silhouette_score(X_reduced,cluster_labels_clean)

# 3. Clustering on Noisy Data (Euclidean Distance)
print("Clustering on Noisy Data (Euclidean Distance)")
cluster_labels_noisy = cluster_and_visualize(X_reduced_noisy, distance_metric='euclidean')
euclidean_noisy_silhoute_score  = silhouette_score(X_reduced,cluster_labels_noisy)

# 4. Clustering on Clean Data (Manhattan Distance)
print("Clustering on Clean Data (Manhattan Distance)")
cluster_labels_clean_manhattan = cluster_and_visualize(X_reduced, distance_metric='manhattan')
manhattan_clean_silhoute_score  = silhouette_score(X_reduced,cluster_labels_clean_manhattan)


print("Silhoute Scores \n Euclidean Distance Without Noise :" ,euclidean_clean_silhoute_score, "\nEuclidean Distance With Noise : ",euclidean_noisy_silhoute_score,"\nManhattan Distance Without Noise : " ,manhattan_clean_silhoute_score)
# 5. Clustering on Noisy Data (Manhattan Distance)
print("Clustering on Noisy Data (Manhattan Distance)")
cluster_labels_noisy_manhattan = cluster_and_visualize(X_reduced_noisy, distance_metric='manhattan')
'''
# 6. Clustering on Clean Data (Cosine Similarity)
print("Clustering on Clean Data (Cosine Similarity)")
cluster_labels_clean_cosine = cluster_and_visualize(X_reduced, distance_metric='cosine')

# 7. Clustering on Noisy Data (Cosine Similarity)
print("Clustering on Noisy Data (Cosine Similarity)")
cluster_labels_noisy_cosine = cluster_and_visualize(X_reduced_noisy, distance_metric='cosine')
'''


#working with the new image
new_image_path = "Samuel_Etoo_0004.jpg"  # Replace with your new image path
def resize_image(image, target_size=(250, 250)):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize(target_size)
    return np.array(resized_image)

new_image = cv2.imread(new_image_path)
new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
resized_image = resize_image(new_image)
flattened_image = resized_image.flatten()
# Preprocess the new image features in the same way (using PCA and scaling)
new_image_scaled = scaler.transform([flattened_image])  # Scale the new image
new_image_pca = pca2.transform(new_image_scaled)  # Apply PCA

X_combined = np.vstack([X_reduced, new_image_pca])

# Fit the HDBSCAN model on the combined dataset (including the new image)
clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric='euclidean',cluster_selection_method='leaf')
new_cluster_labels = clusterer.fit_predict(X_combined)
euclidean_after_silhoute_score  = silhouette_score(X_combined,new_cluster_labels)

print("Silhoute after Adding new Image : ",euclidean_after_silhoute_score)
new_image_cluster = new_cluster_labels[-1]
print(f"Cluster label for the new image: {new_image_cluster}")


# Visualizing the new image's cluster
def visualize_new_image_in_cluster(new_image_pca, clusterer, X_reduced):
    # Ensure new_image_pca is 2D (e.g., [x, y]) for visualization
    if new_image_pca.ndim == 1:
        new_image_pca = new_image_pca.reshape(1, -1)

    # Combine the new image PCA with the existing data
    X_combined = np.vstack([X_reduced, new_image_pca])

    # Re-run HDBSCAN fit_predict on the combined data
    new_cluster_labels = clusterer.fit_predict(X_combined)

    # The label for the new image is the last element in the resulting labels
    new_image_cluster = new_cluster_labels[-1]

    # Displaying the result
    print(f"Cluster label for the new image: {new_image_cluster}")

    # Extract coordinates for the new image
    new_image_coordinates = new_image_pca[0]  # First (and only) point in new_image_pca

    # Plot the data and new image
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=new_cluster_labels[:-1], cmap='viridis', label='Existing Images')
    plt.scatter(new_image_coordinates[0], new_image_coordinates[1], c='red', marker='x', s=100,
                label=f'New Image (Cluster {new_image_cluster})')

    plt.title(f"New Image Cluster: {new_image_cluster}")
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.show()


# Call the function with your data
visualize_new_image_in_cluster(new_image_pca, clusterer, X_reduced)

combined_cluster_labels = clusterer.fit_predict(X_combined)

num_clusters = len(np.unique(combined_cluster_labels[:-1]))
cluster_centers = []
representative_images = []
representative_index = []
# Loop over each cluster
for i in range(num_clusters):
    # Get the points for the current cluster
    cluster_points = X_reduced[combined_cluster_labels[:-1] == i]

    # Skip empty clusters
    if len(cluster_points) == 0:
        continue

    # Calculate the center of the cluster
    center = np.mean(cluster_points, axis=0)
    cluster_centers.append(center)

    # Calculate the distance from each point in the cluster to the cluster center
    distances = euclidean_distances(cluster_points, center.reshape(1, -1))
    closest_point_idx = np.argmin(distances)
    representative_index.append(closest_point_idx)


    # Add the representative image to the list
    representative_images.append(cluster_points[closest_point_idx])

# Display the representative images for each cluster
print("Representative images : ",representative_images)
print("Representative indices : ",representative_index)
fig, ax = plt.subplots(1, len(representative_index), figsize=(10, 6))
for i, index in enumerate(representative_index):
    original_image = images[index]
    # Ensure the images are resized to 250x250 before flattening
    resized_image = resize_image(original_image, target_size=(250, 250))
    ax[i].imshow(resized_image, cmap='gray') # Show grayscale images
    ax[i].set_title(labels[index])
    ax[i].axis('off')
plt.show()

# Save the trained model
joblib.dump(clusterer, 'hdbscan_model.pkl')
print("Model saved!")