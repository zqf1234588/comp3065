from stitching import Stitcher, AffineStitcher
import cv2
import glob
import os

# Get all extracted frames in order
frames_paths = sorted(glob.glob("enhanced_frames/*.png"))

# Create a Stitcher object with custom settings
# You can adjust these settings based on your needs
# settings = {
#     "detector": "sift",  # Feature detector to use (sift, orb, etc.)
#     "confidence_threshold": 0.01,  # Lower value for more image matches
#     "matches_graph_dot_file": "matches_graph.dot",  # Save matches graph for debugging
#     "estimator": "homography",  # Transformation model
#     # "wave_correct": "horiz",  # Wave correction for panorama
#     # "warp": "spherical"  # Warping method (cylindrical, spherical, etc.)
# }
settings = {# The whole plan should be considered
            "crop": True,
            # The matches confidences aren't that good
            "confidence_threshold": 0.05}   

# Create a Stitcher object
stitcher = AffineStitcher(**settings)

# Stitch the frames
panorama = stitcher.stitch(frames_paths)

# Save the result
cv2.imwrite("panorama_result.jpg", panorama)
print("Panorama created successfully!")