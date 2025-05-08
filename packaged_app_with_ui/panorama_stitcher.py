from stitching import Stitcher, AffineStitcher
import cv2
import glob
import os
import uuid
import tempfile
import numpy as np
import math

def create_panorama(
    input_dir, output_file, settings=None, callback=None, interim_callback=None
):
    """
    Create panorama image from images in specified directory

    Parameters:
        input_dir: Directory containing input images
        output_file: Output panorama image file path
        settings: Dictionary of stitcher settings
        callback: Callback function for logging process and updating progress
        interim_callback: Interim result callback for returning real-time results during stitching

    Returns:
        panorama: Stitched panorama image
    """
    # Default settings
    default_settings = {
        "crop": True,
        "confidence_threshold": 0.05,
    }

    # Use provided settings or default settings
    if settings is None:
        settings = default_settings
    else:
        for key, value in default_settings.items():
            if key not in settings:
                settings[key] = value

    if callback:
        callback(f"Starting panorama creation, input directory: {input_dir}")
        callback(f"Using settings: {settings}")

    # Get all extracted frames (in order)
    frames_paths = sorted(glob.glob(f"{input_dir}/*.jpg"))
    if not frames_paths:
        frames_paths = sorted(glob.glob(f"{input_dir}/*.png"))

    if not frames_paths:
        if callback:
            callback(f"Error: No image files found in {input_dir}")
        return None

    if callback:
        callback(f"Found {len(frames_paths)} image files for stitching")

    # If only one image, return it directly
    if len(frames_paths) == 1:
        if callback:
            callback("Only one image, no stitching needed")
        image = cv2.imread(frames_paths[0])
        cv2.imwrite(output_file, image)
        if interim_callback:
            interim_callback(image, 1, 1)
        return image

    # Read all images
    images = []
    for path in frames_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)

    if len(images) == 0:
        if callback:
            callback("Error: Unable to read any images")
        return None

    # Create stitcher object
    if settings.get("estimator") == "affine":
        if callback:
            callback("Using affine transform stitcher")
        stitcher = AffineStitcher(**settings)
    else:
        if callback:
            callback("Using standard stitcher")
        stitcher = Stitcher(**settings)

    # Stitch all images at once
    if callback:
        callback("Stitching all images...")

    panorama = stitcher.stitch(images)

    if panorama is None or panorama.size == 0:
        if callback:
            callback("Stitching failed: Unable to create panorama")
        return None

    # Save result
    cv2.imwrite(output_file, panorama)
    if callback:
        callback(f"Panorama successfully created: {output_file}")
    if interim_callback:
        interim_callback(panorama, len(images), len(images))
    return panorama


def stitch_panorama(
    frames_dir, output_path, settings=None, callback=None, interim_callback=None
):
    """
    Create panorama from specified directory (app.py compatible version)

    Parameters:
        frames_dir: Directory containing frames
        output_path: Output panorama path
        settings: Stitching settings
        callback: Callback function
        interim_callback: Interim result callback function

    Returns:
        output_path: Returns output path if successful; otherwise None
    """

    # Wrap callback function to adapt to create_panorama function
    def wrapped_callback(message):
        if callback:
            callback(message)

    # Try to create panorama
    panorama = create_panorama(
        input_dir=frames_dir,
        output_file=output_path,
        settings=settings,
        callback=wrapped_callback,
        interim_callback=interim_callback,
    )

    # If creation successful, return output path; otherwise return None
    if panorama is not None:
        return output_path
    return None


def main():
    # Usage example
    input_dir = "frames1"
    output_file = f"panorama_{uuid.uuid4().hex[:8]}.jpg"

    # Define callback function
    def print_callback(message):
        print(f"[Panorama stitcher] {message}")

    # Define interim result callback
    def interim_result_callback(image, current, total):
        temp_file = f"temp_panorama_{current}_{total}.jpg"
        cv2.imwrite(temp_file, image)
        print(
            f"[Panorama stitcher] Saved interim result: {temp_file}, progress: {current}/{total}"
        )

    # Custom settings
    settings = {
        "detector": "sift",
        "confidence_threshold": 0.05,
        "matches_graph_dot_file": "matches_graph.dot",
        "estimator": "homography",
        "crop": True,
    }

    # Create panorama
    panorama = create_panorama(
        input_dir=input_dir,
        output_file=output_file,
        settings=settings,
        callback=print_callback,
        interim_callback=interim_result_callback,
    )

    if panorama is not None:
        print(f"Panorama created: {output_file}")
    else:
        print("Panorama creation failed")


if __name__ == "__main__":
    main()
