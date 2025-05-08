# Comp3065 CW topic A Project

This project processes video frames and generates high-quality panorama images using **Super-Resolution (Real-ESRGAN)** and an image **stitching toolkit**.

## 📁 Project Structure

Panorama/
├── frames(normal)/ # Frame images extracted from video 1

├── frames(slightlyblurry)/ # Frame images extracted from video 2

├── frames(veryblurry)/ # Frame images extracted from video 3

├── panorama_result(normal).jpg # Panorama result from video 1

├── panorama_result(slightlyblurry).jpg # Panorama result from video 2

├── panorama_result(veryblurry).jpg # Panorama result from video 3

├── test_input1(normal).mp4 # Input video 1

├── test_input2(slightlyblurry).mp4 # Input video 2

├── test_input3(veryblurry).mp4 # Input video 3

├── packaged_app_with_ui #This folder provides a fully packaged and user-friendly version of the application, with core functionalities wrapped in callable interfaces and integrated with a graphical user interface (UI).

├── packaged_app_with_ui #This folder provides a fully packaged and user-friendly version of the application, with core functionalities wrapped in callable interfaces and integrated with a graphical user interface (UI).

├── README.md # documentation of project structure



## External Techniques

-  **Panorama Stitching Toolkit**  
  GitHub: [OpenStitching/stitching](https://github.com/OpenStitching/stitching/tree/main)

-  **Super-Resolution via Real-ESRGAN**  
  GitHub: [xinntao/Real-ESRGAN - inference script](https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py)



