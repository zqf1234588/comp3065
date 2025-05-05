# Comp3065 CW topic A Project

This project processes video frames and generates high-quality panorama images using **Super-Resolution (Real-ESRGAN)** and an image **stitching toolkit**.

## 📁 Project Structure

Panorama/
├── enhanced_frames/ # Output folder: super-resolved frames using Real-ESRGAN

├── frames(normal)/ # Frame images extracted from video 1

├── frames(slightlyblurry)/ # Frame images extracted from video 2

├── frames(veryblurry)/ # Frame images extracted from video 3

├── Real-ESRGAN/ # Real-ESRGAN model and source code

├── stitching/ # Panorama stitching utility library

├── weights/ # Super-resolution model weights

├── matches_graph.dot # Keypoint matching graph

├── panorama_result(normal).jpg # Panorama result from video 1

├── panorama_result(slightlyblurry).jpg # Panorama result from video 2

├── panorama_result(veryblurry).jpg # Panorama result from video 3

├── step1.py # Extract frames and apply super-resolution

├── step2.py # Stitch images into panoramas

├── test1(normal).mp4 # Input video 1

├── test2(slightlyblurry).mp4 # Input video 2

├── test3(veryblurry).mp4 # Input video 3

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

