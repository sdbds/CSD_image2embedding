import os
import shutil
from cluster_result_utils import get_cluster_labels, get_noise_label, has_noise_cluster


def classify_images(data, clustering_result, args, output_dir=None):
    if output_dir is None:
        output_dir = args
        symlink = False
    else:
        symlink = bool(getattr(args, "symlink", False))

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get cluster labels and image data
    labels = clustering_result.labels_
    paths = data["path"].tolist()

    # Create a subdirectory for each cluster
    for label in get_cluster_labels(clustering_result):
        cluster_dir = os.path.join(output_dir, f"class_{label}")
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

    noise_dir = None
    noise_label = get_noise_label(clustering_result)
    if has_noise_cluster(clustering_result):
        noise_dir = os.path.join(output_dir, "noise")
        if not os.path.exists(noise_dir):
            os.makedirs(noise_dir)

    # Copy images to their respective cluster directories
    for i, (label, image_abs_path) in enumerate(zip(labels, paths)):
        if noise_label is not None and label == noise_label:
            target_dir = noise_dir
        else:
            target_dir = os.path.join(output_dir, f"class_{label}")

        image_path = os.path.join(target_dir, f"image_{i}.jpg")

        if not os.path.exists(image_abs_path):
            print(f"Warning: {image_abs_path} does not exist and will be skipped.")
            continue
        if os.path.exists(image_path):
            print(f"Warning: {image_path} already exists and will be skipped.")
            continue

        if symlink:
            os.symlink(image_abs_path, image_path)
        else:
            shutil.copy(image_abs_path, image_path)

    print(f"Images have been classified and saved to {output_dir}")
