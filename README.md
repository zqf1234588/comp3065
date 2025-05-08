# Comp3065 CW topic A Project

This project processes video frames and generates high-quality panorama images using **Super-Resolution (Real-ESRGAN)** and an image **stitching toolkit**.

# Declaration of this coursework
The `core_code_analysis folder` contains the low-level implementation of the system, extracted and organized separately to improve readability and transparency. This structure helps demonstrate that I fully understand the internal mechanisms of the third-party libraries used, and have not simply copied their functionality. Key algorithms and logic are explained in detail, and their underlying principles will be documented in the accompanying report.

The `packaged_app_with_ui` folder contains the final application with a complete user interface. It includes features such as video upload, frame extraction (a simplified version of the core logic), and optional super-resolution processing. Multiple super-resolution models are supported and can be selected or disabled via the UI. Unlike the fixed parameters in core_code_analysis, the user interface in this version allows dynamic control over parameters such as feature detector type, confidence threshold, and transformation model for panorama generation.
## 📁 Project Structure
```python
/
├── frames(normal)/ # Frame images extracted from video 1
├── frames(slightlyblurry)/ # Frame images extracted from video 2
├── frames(veryblurry)/ # Frame images extracted from video 3
├── panorama_result(normal).jpg # Panorama result from video 1
├── panorama_result(slightlyblurry).jpg # Panorama result from video 2
├── panorama_result(veryblurry).jpg # Panorama result from video 3
├── test_input1(normal).mp4 # Input video 1
├── test_input2(slightlyblurry).mp4 # Input video 2
├── test_input3(veryblurry).mp4 # Input video 3
├── packaged_app_with_ui #This folder provides a fully packaged and user-friendly version of the application with core functionalities wrapped in callable interfaces and integrated with a graphical user interface (UI).
├── packaged_app_with_ui #This folder provides a fully packaged and user-friendly version of the application, with core functionalities wrapped in callable interfaces and integrated with a graphical user interface (UI).
├── README.md # documentation of project structure
```


## External Techniques

-  **Panorama Stitching Toolkit**  
  GitHub: [OpenStitching/stitching](https://github.com/OpenStitching/stitching/tree/main)

-  **Super-Resolution via Real-ESRGAN**  
  GitHub: [xinntao/Real-ESRGAN - inference script](https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py)

  



