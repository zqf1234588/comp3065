# Comp3065 CW topic A Project

This project processes video frames and generates high-quality panorama images using **Super-Resolution (Real-ESRGAN)** and an image **stitching toolkit**.

## 📁 Project Structure

Panorama/

├── Real-ESRGAN/ # Real-ESRGAN model and source code

├── stitching/ # Panorama stitching utility library

├── weights/ # Super-resolution model weights

├── matches_graph.dot # Keypoint matching graph


├── step1.py # Extract frames and apply super-resolution

├── step2.py # Stitch images into panoramas

├── README.md # documentation of project structure



## External Techniques

-  **Panorama Stitching Toolkit**  
  GitHub: [OpenStitching/stitching](https://github.com/OpenStitching/stitching/tree/main)

-  **Super-Resolution via Real-ESRGAN**  
  GitHub: [xinntao/Real-ESRGAN - inference script](https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py)

##  Usage

1. **Run `step1.py`**  
   - Extracts frames from input videos  
   - Applies Real-ESRGAN to enhance resolution

2. **Run `step2.py`**  
   - Matches keypoints(features) between frames  
   - Generates panorama images from enhanced frames

