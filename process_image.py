import os
import shutil
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN

def classify_images(data, kmeans_result, args, output_dir):

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get cluster labels and image data
    labels = kmeans_result.labels_
    paths = data["path"].tolist()

    # Get the number of clusters
    if isinstance(kmeans_result, KMeans):
        n_clusters = kmeans_result.n_clusters
    else:  # HDBSCAN
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Create a subdirectory for each cluster
    for i in range(n_clusters):
        cluster_dir = os.path.join(output_dir, f"class_{i}")
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

    # Create a directory for noise points (only for HDBSCAN)
    if isinstance(kmeans_result, HDBSCAN):
        noise_dir = os.path.join(output_dir, "noise")
        if not os.path.exists(noise_dir):
            os.makedirs(noise_dir)

    # Copy images to their respective cluster directories
    for i, (label, image_abs_path) in enumerate(zip(labels, paths)):
        if label == -1:  # Noise point (HDBSCAN only)
            target_dir = noise_dir
        else:
            target_dir = os.path.join(output_dir, f"class_{label}")
        
        image_path = os.path.join(target_dir, f"image_{i}.jpg")
        
        # Copy image to the target directory
        if os.path.exists(image_abs_path) and not os.path.exists(image_path):
            if args.symlink:
                os.symlink(image_abs_path, image_path)
            else:
                shutil.copy(image_abs_path, image_path)
        elif not os.path.exists(image_path):
            print(f"Warning: {image_path} already exists and will be skipped.")
        else:
            print(f"Warning: {image_abs_path} does not exist and will be skipped.")

    print(f"Images have been classified and saved to {output_dir}")