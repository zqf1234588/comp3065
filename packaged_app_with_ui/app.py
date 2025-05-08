import gradio as gr
import os
import cv2
import tempfile
import shutil
import time
import uuid
from pathlib import Path
import glob
import atexit

# Import other modules
import frame_processor
import panorama_stitcher

# Store all created temporary directories for cleanup on exit
TEMP_DIRS = []


def cleanup_temp_dirs():
    """Clean up all temporary directories"""
    global TEMP_DIRS
    for dir_path in TEMP_DIRS:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Cleaned up temporary directory: {dir_path}")
            except Exception as e:
                print(f"Failed to clean up temporary directory {dir_path}: {str(e)}")
    TEMP_DIRS.clear()


# Register cleanup function for exit
atexit.register(cleanup_temp_dirs)


def process_video_to_panorama(
    video_path,
    frame_skip,
    enhance_enabled,
    model_name,
    outscale,
    crop,
    detector,
    confidence_threshold,
    estimator,
    progress=gr.Progress(),
):
    """Process video and create panorama"""
    global TEMP_DIRS

    if not video_path:
        return None

    # Create working directory
    work_dir = f"process_{uuid.uuid4().hex[:8]}"
    os.makedirs(work_dir, exist_ok=True)
    TEMP_DIRS.append(work_dir)

    # Step 1: Extract frames
    progress(0, desc="Preparing to process video...")

    # Process video parameter settings
    params = {
        "frame_skip": frame_skip,
        "enhance": enhance_enabled,
        "model_name": model_name,
        "outscale": outscale,
    }

    # Process video
    result = frame_processor.process_video(
        video_path=video_path,
        output_dir=work_dir,
        params=params,
        callback=lambda msg, prog=None: progress(
            prog * 0.7 if prog is not None else 0.3, desc=msg
        ),
    )

    if result["frames_count"] == 0:
        return None

    # Get enhanced frames directory or original frames directory
    frames_dir = result.get("enhanced_dir", result["frames_dir"])

    # Step 2: Stitch panorama
    progress(0.7, desc="Stitching panorama...")
    output_path = os.path.join(work_dir, "panorama.jpg")

    # Stitching settings
    panorama_settings = {"crop": crop}

    # Set confidence threshold
    try:
        confidence_val = (
            float(confidence_threshold.strip())
            if confidence_threshold.strip()
            else 0.05
        )
    except ValueError:
        confidence_val = 0.05
        progress(
            0.7, desc="Invalid confidence threshold format, using default value 0.05"
        )

    panorama_settings["confidence_threshold"] = confidence_val
    panorama_settings["detector"] = detector if detector else "sift"
    panorama_settings["estimator"] = estimator if estimator else "homography"

    # Stitch panorama
    panorama = panorama_stitcher.create_panorama(
        input_dir=frames_dir,
        output_file=output_path,
        settings=panorama_settings,
        callback=lambda msg: progress(0.8, desc=msg),
        interim_callback=lambda img, curr, total: progress(
            0.7 + 0.3 * curr / total, desc=f"Stitching progress: {curr}/{total}"
        ),
    )

    progress(1.0, desc="Processing complete!")

    # Save result and return
    if panorama is None:
        return None

    # Copy result to a temporary file that won't be automatically deleted
    final_output = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    cv2.imwrite(final_output, panorama)

    return final_output


def get_available_models():
    """Get list of available models"""
    if hasattr(frame_processor, "AVAILABLE_MODELS"):
        return [model["name"] for model in frame_processor.AVAILABLE_MODELS]
    return [
        "RealESRGAN_x4plus",
        "RealESRGAN_x4plus_anime_6B",
        "RealESRGAN_x2plus",
        "RealESRNet_x4plus",
        "realesr-animevideov3",
        "realesr-general-x4v3",
    ]


def create_ui():
    """Create Gradio user interface"""
    with gr.Blocks(title="Panorama Creator") as app:
        gr.Markdown("# Panorama Creator")
        gr.Markdown(
            "Upload video, extract frames, enhance image quality, then create panorama."
        )

        # Top - Video input and parameters
        with gr.Row():
            # Left - Video input
            with gr.Column(scale=1):
                video_input = gr.Video(label="Input Video")

            # Right - Parameter controls
            with gr.Column(scale=1):
                # Basic parameters
                frame_skip = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=5,
                    step=1,
                    label="Frame Skip (higher = faster processing but less detail)",
                )

                # Super-resolution settings
                with gr.Accordion("Super-Resolution Settings", open=True):
                    enhance_enabled = gr.Checkbox(
                        label="Enable Super-Resolution", value=True
                    )
                    model_name = gr.Dropdown(
                        choices=get_available_models(),
                        value="RealESRGAN_x2plus",
                        label="Super-Resolution Model",
                    )
                    outscale = gr.Slider(
                        minimum=1,
                        maximum=4,
                        value=2,
                        step=1,
                        label="Output Scale Factor",
                    )

                # Panorama settings
                with gr.Accordion("Panorama Stitching Settings", open=True):
                    crop = gr.Checkbox(label="Crop Edges", value=True)
                    detector = gr.Dropdown(
                        choices=["sift", "orb"], value="sift", label="Feature Detector"
                    )
                    confidence_threshold = gr.Textbox(
                        label="Confidence Threshold (0-1)",
                        value="0.05",
                        info="Lower values match more images",
                    )
                    estimator = gr.Dropdown(
                        choices=["homography", "affine"],
                        value="homography",
                        label="Transform Model",
                    )

                # Process button
                process_btn = gr.Button("Process Video", variant="primary")

        # Bottom - Panorama result (full row)
        with gr.Row():
            panorama_output = gr.Image(label="Panorama Result", type="filepath")

        # Process button click event
        process_btn.click(
            fn=process_video_to_panorama,
            inputs=[
                video_input,
                frame_skip,
                enhance_enabled,
                model_name,
                outscale,
                crop,
                detector,
                confidence_threshold,
                estimator,
            ],
            outputs=panorama_output,
        )

        # Add usage tips
        with gr.Accordion("Usage Instructions", open=False):
            gr.Markdown(
                """
                ### How to Use
                1. Upload a video file (preferably with horizontal or vertical panning)
                2. Adjust frame skip value (lower values extract more frames, longer processing time)
                3. Choose whether to enable super-resolution and set related parameters
                4. Adjust panorama stitching settings
                5. Click "Process Video" button to start processing
                
                ### Execution Flow
                - The system extracts video frames to a temporary folder
                - If super-resolution is enabled, all extracted frames are enhanced
                - Finally, all frames are stitched together to create a panorama
                
                ### Notes
                - Videos with panning motion work best for panorama creation
                - Processing time depends on video length, frame skip, and computer performance
                - Lowering the confidence threshold matches more images but may cause incorrect matches
                """
            )

        # Add examples (if example video exists)
        if os.path.exists("example_video.mp4"):
            gr.Examples(
                examples=[
                    [
                        "example_video.mp4",
                        5,
                        True,
                        "RealESRGAN_x4plus",
                        2,
                        True,
                        "sift",
                        "0.05",
                        "homography",
                    ]
                ],
                inputs=[
                    video_input,
                    frame_skip,
                    enhance_enabled,
                    model_name,
                    outscale,
                    crop,
                    detector,
                    confidence_threshold,
                    estimator,
                ],
            )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False)
