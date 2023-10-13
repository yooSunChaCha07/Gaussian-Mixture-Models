#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment 4: Gaussian Mixture Models
Yoo Sun Song
301091906
"""

# load the Olivetti faces dataset
from sklearn.datasets import fetch_olivetti_faces

olivetti_faces = fetch_olivetti_faces(shuffle=True, random_state=96)
X = olivetti_faces.data

print('X Shape: ', X.shape)


#1. Use PCA preserving 99% of the variance to reduce the dataset’s dimensionality. 
from sklearn.decomposition import PCA

pca = PCA(n_components=0.99, whiten=True)
X_pca = pca.fit_transform(X)

print('X Shape: ', X_pca.shape)


#2.	Determine the most suitable covariance_type for the dataset.
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Define the types of covariance matrices 
covariance_types = ['full', 'tied', 'diag', 'spherical']

# Lists to store the AIC and BIC values for each covariance type
aic_values = []
bic_values = []

# Loop through each covariance type
for cov_type in covariance_types:
    # Initialize a Gaussian Mixture Model with 10 components and the given covariance type
    gmm = GaussianMixture(n_components=10, covariance_type=cov_type, random_state=96)
    # Fit the GMM to the PCA-transformed data
    gmm.fit(X_pca)
     # Append the AIC and BIC values of the fitted model to the respective lists
    aic_values.append(gmm.aic(X_pca))
    bic_values.append(gmm.bic(X_pca))

    # Print the results for the current covariance type
    print('\nCovariance_type: ', cov_type)
    print('AIC: ', gmm.aic(X_pca))
    print('BIC: ', gmm.bic(X_pca))
    print('------------------------------')

import numpy as np

# Rank covariance types based on AIC and BIC values

# argsort() provides the indices that would sort the list.
aic_ranks = np.argsort(aic_values)
bic_ranks = np.argsort(bic_values)

# Define colors for the bars in the plot
colors = ['red', 'blue', 'green', 'yellow'] 
bar_width = 0.4

# Set the figure size for the plots
plt.figure(figsize=(15,7))

# Plot AIC values
plt.subplot(1, 2, 1)  # subplot with 1 row and 2 columns, activating the 1st plot
for idx, rank in enumerate(aic_ranks):
    # Create a bar for each covariance type, colored based on its rank
    plt.bar(idx, aic_values[rank], width=bar_width, label=f'{covariance_types[rank]}', color=colors[rank], alpha=0.7)
    # Annotate each bar with the AIC value
    plt.text(idx, aic_values[rank] + 5, round(aic_values[rank], 2), ha='center', va='bottom')

# Set labels, title, x-ticks, and grid for the AIC plot
plt.xlabel('Rank')
plt.ylabel('AIC Value')
plt.title('AIC values ranked for different Covariance Types')
plt.xticks(range(len(covariance_types)), range(1, len(covariance_types)+1))
plt.legend()
plt.grid(axis='y')

# Plot BIC values
plt.subplot(1, 2, 2)  # activating the 2nd plot
for idx, rank in enumerate(bic_ranks):
    # Create a bar for each covariance type, colored based on its rank
    plt.bar(idx, bic_values[rank], width=bar_width, label=f'{covariance_types[rank]}', color=colors[rank], alpha=0.7)
    # Annotate each bar with the BIC value
    plt.text(idx, bic_values[rank] + 5, round(bic_values[rank], 2), ha='center', va='bottom')

# Set labels, title, x-ticks, and grid for the BIC plot
plt.xlabel('Rank')
plt.ylabel('BIC Value')
plt.title('BIC values ranked for different Covariance Types')
plt.xticks(range(len(covariance_types)), range(1, len(covariance_types)+1))
plt.legend()
plt.grid(axis='y')

# Adjust the layout to ensure plots don't overlap
plt.tight_layout()

# Display the plots
plt.show()


#Determines the best covariance type for Gaussian Mixture Models based on AIC and BIC values

# List of covariance types to evaluate
covariance_types = ['full', 'tied', 'diag', 'spherical']

# Initialize variables to store the lowest AIC and BIC values
lowest_aic = float('inf')
lowest_bic = float('inf')

# Variables to store the best covariance type for AIC and BIC
best_covariance_types_aic = None
best_covariance_types_bic = None

# Loop through each covariance type to train the GMM and compute AIC and BIC
for cov_type in covariance_types:
    # Initialize and train the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=10, covariance_type=cov_type, random_state=96)
    gmm.fit(X_pca)

    # Compute AIC and BIC for the trained model
    aic = gmm.aic(X_pca)
    bic = gmm.bic(X_pca)

    # Update the lowest AIC and associated covariance type if the current AIC is lower
    if aic < lowest_aic:
        lowest_aic = aic
        best_covariance_types_aic = cov_type

    # Update the lowest BIC and associated covariance type if the current BIC is lower
    if bic < lowest_bic:
        lowest_bic = bic
        best_covariance_types_bic = cov_type

# Print the results: the best covariance type for both AIC and BIC
print(f"Best covariance type based on AIC: {best_covariance_types_aic}")
print(f"Best covariance type based on BIC: {best_covariance_types_bic}")


#3.	Determine the minimum number of clusters that best represent the dataset using either AIC or BIC. 

# Using AIC to determine the best number of clusters

# Define the range of possible number of clusters to evaluate
n_components_range = range(1, 50)

# Initialize variable to store the lowest AIC value
lowest_aic = float('inf')

# Variable to store the best number of clusters based on AIC
best_n_clusters_aic = None

# List to store AIC values for each number of clusters
aics = []

# Loop through each possible number of clusters to train the GMM and compute AIC
for n_clusters in n_components_range:
    # Initialize and train the Gaussian Mixture Model with the best covariance type determined previously for AIC
    gmm = GaussianMixture(n_components=n_clusters, covariance_type=best_covariance_types_aic, random_state=96)
    gmm.fit(X_pca)

    # Compute AIC for the trained model
    aic = gmm.aic(X_pca)
    # Append the AIC value to the list
    aics.append(aic)
    
    # Update the lowest AIC and associated number of clusters if the current AIC is lower
    if aic < lowest_aic:
        lowest_aic = aic
        best_n_clusters_aic = n_clusters

# Print the results: the best number of clusters for AIC
print(f"Best number of clusters according to AIC: {best_n_clusters_aic}")

# Using BIC to determine the best number of clusters

# Initialize variable to store the lowest BIC value
lowest_bic = float('inf')

# Variable to store the best number of clusters based on BIC
best_n_clusters_bic = None

# List to store BIC values for each number of clusters
bics = []

# Loop through each possible number of clusters to train the GMM and compute BIC
for n_clusters in n_components_range:
    # Initialize and train the Gaussian Mixture Model with the best covariance type determined previously for BIC
    gmm = GaussianMixture(n_components=n_clusters, covariance_type=best_covariance_types_bic, random_state=96)
    gmm.fit(X_pca)

    # Compute BIC for the trained model
    bic = gmm.bic(X_pca)
    # Append the BIC value to the list
    bics.append(bic)
    
    # Update the lowest BIC and associated number of clusters if the current BIC is lower
    if bic < lowest_bic:
        lowest_bic = bic
        best_n_clusters_bic = n_clusters

# Print the results: the best number of clusters for BIC
print(f"Best number of clusters according to BIC: {best_n_clusters_bic}")


#4.	Plot the results from (2) and (3)
plt.figure(figsize=(14, 6))
plt.plot(n_components_range, aics, label='AIC', marker='o')
plt.plot(n_components_range, bics, label='BIC', marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Information Criterion')
plt.legend(loc ='best')
plt.title('AIC and BIC for Different Cluster Numbers')
plt.grid(True)

# For visual interpretation
min_aic = min(aics)
min_bic = min(bics)
elbow_aic = n_components_range[aics.index(min_aic)]
elbow_bic = n_components_range[bics.index(min_bic)]

plt.axvline(x=elbow_aic, color='blue', linestyle='--', label=f"Elbow AIC at {elbow_aic} clusters")
plt.axvline(x=elbow_bic, color='red', linestyle='--', label=f"Elbow BIC at {elbow_bic} clusters")
plt.legend()

plt.show()

#5.	Output the hard clustering for each instance. 
#6.	Output the soft clustering for each instance.

# Using the BIC optimal number of clusters

# Initialize and train the Gaussian Mixture Model with the best number of clusters based on BIC
gmm_best_bic = GaussianMixture(n_components=best_n_clusters_bic, covariance_type=best_covariance_types_bic, random_state=96)
gmm_best_bic.fit(X_pca)

# Predict the cluster label for each instance in the data (Hard clustering)
hard_clusters_bic = gmm_best_bic.predict(X_pca)

# Predict the probabilities of each instance belonging to each cluster (Soft clustering)
soft_clusters_bic = gmm_best_bic.predict_proba(X_pca)

# Print the hard clustering results
print("Hard clustering labels for each instance (Using BIC):", hard_clusters_bic)

# Print the soft clustering results
print("Soft clustering probabilities for each instance (Using BIC):", soft_clusters_bic)

# Using the AIC optimal number of clusters

# Initialize and train the Gaussian Mixture Model with the best number of clusters based on AIC
gmm_best_aic = GaussianMixture(n_components=best_n_clusters_aic, covariance_type=best_covariance_types_aic, random_state=96)
gmm_best_aic.fit(X_pca)

# Predict the cluster label for each instance in the data (Hard clustering)
hard_clusters_aic = gmm_best_aic.predict(X_pca)

# Predict the probabilities of each instance belonging to each cluster (Soft clustering)
soft_clusters_aic = gmm_best_aic.predict_proba(X_pca)

# Print the hard clustering results
print("Hard clustering labels for each instance (Using AIC):", hard_clusters_aic)

# Print the soft clustering results
print("Soft clustering probabilities for each instance (Using AIC):", soft_clusters_aic)


#7.	Use the model to generate some new faces (using the sample() method),and visualize them 
#(use the inverse_transform() method to transform the data back to its original space based on the PCA method used). 

# Number of new faces to be generated
num_new_faces = 15

# Use the sample() method of the Gaussian Mixture Model to generate new samples (new faces in PCA space) Using BIC
new_samples, _ = gmm_best_bic.sample(n_samples=num_new_faces)

# Transform the samples from PCA space back to the original space using the inverse_transform() method of the PCA
new_faces = pca.inverse_transform(new_samples)

# Plotting the generated faces
plt.figure(figsize=(15, 9))

# Loop through the generated faces and plot them
for i, face in enumerate(new_faces):
    plt.subplot(3, 5, i + 1)  
    plt.imshow(face.reshape(64, 64), cmap='bone')  # Reshape the face data to 64x64 and display as an image
    plt.axis('off')  # Hide the axis

plt.tight_layout()

plt.show()


#8.	Modify some images (e.g., rotate, flip, darken). 
from scipy.ndimage import rotate
import cv2

# Rotate the new faces generated by 45 degrees without reshaping them
rotated_images = [rotate(image.reshape(64, 64), 45, reshape=False, mode='nearest') for image in new_faces]

# Plot the rotated images in a 3x5 grid
plt.figure(figsize=(15, 9))
for index, face in enumerate(rotated_images):
    plt.subplot(3, 5, index + 1)
    plt.imshow(face, cmap='bone')
    plt.axis('off')
plt.suptitle('Rotated Generated Faces by 45 Degrees')
plt.tight_layout()
plt.show()

# Flip the new faces horizontally
flipped_images = [np.fliplr(image.reshape(64, 64)) for image in new_faces]

# Plot the flipped images in a 3x5 grid
plt.figure(figsize=(15, 9))
for index, face in enumerate(flipped_images):
    plt.subplot(3, 5, index + 1)
    plt.imshow(face, cmap='bone')
    plt.axis('off')
plt.suptitle('Flipped Generated Faces')
plt.tight_layout()
plt.show()

# Convert the new faces to a 0-255 range for processing
scaled_images = [image.reshape(64, 64) * 255 for image in new_faces]

# Darken the images using cv2's convertScaleAbs method 
# (alpha value less than 1 reduces brightness, and beta value subtracts brightness)
darkened_images = [cv2.convertScaleAbs(image, alpha=0.5, beta=-50) for image in scaled_images]

# Plot the darkened images in a 3x5 grid
plt.figure(figsize=(15, 9))
for index, face in enumerate(darkened_images):
    plt.subplot(3, 5, index + 1)
    plt.imshow(face, cmap='bone')
    plt.axis('off')
plt.suptitle('Darkened Generated Faces using cv2')
plt.tight_layout()
plt.show()

#9.	Determine if the model can detect the anomalies produced in (8) 
#by comparing the output of the score_samples() method for normal images and for anomalies). 

# Compute the log likelihood of each sample in the original images using the best GMM model trained with AIC
original_scores = gmm_best_aic.score_samples(pca.transform(new_faces.reshape(num_new_faces, -1)))

# Flatten the modified images to convert them from 2D (64x64) to 1D (4096) 
rotated_2d = np.array([img.flatten() for img in rotated_images])
flipped_2d = np.array([img.flatten() for img in flipped_images])
darkened_2d = np.array([img.flatten() for img in darkened_images])

# Transform these flattened images back into the PCA space
rotated_pca = pca.transform(rotated_2d)
flipped_pca = pca.transform(flipped_2d)
darkened_pca = pca.transform(darkened_2d)

# Compute the log likelihood of each sample in the modified images using the best GMM model trained with AIC
rotated_scores = gmm_best_aic.score_samples(rotated_pca)
flipped_scores = gmm_best_aic.score_samples(flipped_pca)
darkened_scores = gmm_best_aic.score_samples(darkened_pca)

# Display the average log likelihoods for each set of images
print("Average score for original images:", np.mean(original_scores))
print("Average score for rotated images:", np.mean(rotated_scores))
print("Average score for flipped images:", np.mean(flipped_scores))
print("Average score for darkened images:", np.mean(darkened_scores))

# Compute the log likelihood of each sample in the original images using the best GMM model trained with BIC
original_scores = gmm_best_bic.score_samples(pca.transform(new_faces.reshape(num_new_faces, -1)))

# Flatten the modified images to convert them from 2D (64x64) to 1D (4096) 
rotated_2d = np.array([img.flatten() for img in rotated_images])
flipped_2d = np.array([img.flatten() for img in flipped_images])
darkened_2d = np.array([img.flatten() for img in darkened_images])

# Transform these flattened images back into the PCA space
rotated_pca = pca.transform(rotated_2d)
flipped_pca = pca.transform(flipped_2d)
darkened_pca = pca.transform(darkened_2d)

# Compute the log likelihood of each sample in the modified images using the best GMM model trained with BIC
rotated_scores = gmm_best_bic.score_samples(rotated_pca)
flipped_scores = gmm_best_bic.score_samples(flipped_pca)
darkened_scores = gmm_best_bic.score_samples(darkened_pca)

# Display the average log likelihoods for each set of images
print("Average score for original images:", np.mean(original_scores))
print("Average score for rotated images:", np.mean(rotated_scores))
print("Average score for flipped images:", np.mean(flipped_scores))
print("Average score for darkened images:", np.mean(darkened_scores))



