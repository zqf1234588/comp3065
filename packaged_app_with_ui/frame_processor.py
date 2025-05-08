import cv2
import os
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url
import glob
import uuid


# Define available model list
AVAILABLE_MODELS = [
    {"name": "RealESRGAN_x4plus", "description": "General model, 4x upscale"},
    {
        "name": "RealESRGAN_x4plus_anime_6B",
        "description": "Anime model, 4x upscale, smaller",
    },
    {"name": "RealESRGAN_x2plus", "description": "General model, 2x upscale"},
    {"name": "RealESRNet_x4plus", "description": "Denoising model, 4x upscale"},
    {"name": "realesr-animevideov3", "description": "Anime video model, 4x upscale"},
    {"name": "realesr-general-x4v3", "description": "General video model, 4x upscale"},
]


def create_directory(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def extract_frames(video_path, output_dir, frame_skip=1, callback=None):
    """
    Extract frames from video

    Parameters:
        video_path: Video file path
        output_dir: Output directory
        frame_skip: Frame interval
        callback: Callback function for progress updates

    Returns:
        saved_count: Number of frames saved
    """
    create_directory(output_dir)

    # Open video file
    if callback:
        callback(f"Opening video: {video_path}")

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        if callback:
            callback(f"Error: Unable to open video {video_path}")
        return 0

    # Get video information
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    if callback:
        callback(
            f"Video info: Total frames={total_frames}, FPS={fps:.2f}, Duration={duration:.2f}s"
        )
        callback(f"Starting frame extraction, interval={frame_skip}...")

    count = 0
    saved_count = 0

    while True:
        success, frame = video.read()
        if not success:
            break

        if count % frame_skip == 0:
            frame_path = f"{output_dir}/frame_{count:04d}.jpg"
            cv2.imwrite(frame_path, frame)
            saved_count += 1

            if callback and saved_count % 10 == 0:
                progress = min(count / total_frames, 1.0) if total_frames > 0 else 0
                callback(f"Saved {saved_count} frames", progress)

        count += 1

    video.release()

    if callback:
        callback(f"Frame extraction complete! Extracted {saved_count} frames", 1.0)

    return saved_count


def get_model(model_name):
    """
    Get model configuration based on model name

    Parameters:
        model_name: Model name

    Returns:
        tuple: (model, netscale, file_url, dni_weight)
    """
    model = None
    netscale = 4
    file_url = []
    dni_weight = None

    if model_name == "RealESRGAN_x4plus":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        ]
    elif model_name == "RealESRNet_x4plus":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
        ]
    elif model_name == "RealESRGAN_x4plus_anime_6B":
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        ]
    elif model_name == "RealESRGAN_x2plus":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        netscale = 2
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        ]
    elif model_name == "realesr-animevideov3":
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=16,
            upscale=4,
            act_type="prelu",
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
        ]
    elif model_name == "realesr-general-x4v3":
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        ]

    return model, netscale, file_url, dni_weight


def enhance_frames(input_dir, output_dir, upsampler, outscale=4, callback=None):
    """
    Enhance frames

    Parameters:
        input_dir: Input directory containing original frames
        output_dir: Output directory for enhanced frames
        upsampler: RealESRGANer instance
        outscale: Output scale factor
        callback: Callback function for progress updates

    Returns:
        enhanced_count: Number of enhanced frames
    """
    create_directory(output_dir)

    frame_paths = sorted(glob.glob(f"{input_dir}/frame_*.jpg"))
    total_frames = len(frame_paths)
    enhanced_count = 0

    if callback:
        callback(
            f"Starting enhancement of {total_frames} frames, scale factor: {outscale}..."
        )

    for idx, path in enumerate(frame_paths):
        frame_number = os.path.basename(path).split("_")[1].split(".")[0]
        output_path = os.path.join(output_dir, f"enhanced_{frame_number}.png")

        # Read image
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            if callback:
                callback(f"Unable to read image: {path}")
            continue

        # Enhance
        output, _ = upsampler.enhance(img, outscale=outscale)

        # Save enhanced result
        cv2.imwrite(output_path, output)
        enhanced_count += 1

        # Update progress
        progress = (idx + 1) / total_frames
        if callback and ((idx + 1) % 5 == 0 or idx == 0 or idx == total_frames - 1):
            callback(
                f"Enhancement progress: {idx+1}/{total_frames} ({progress*100:.1f}%)",
                progress,
            )

    if callback:
        callback(f"Frame enhancement complete! Processed {enhanced_count} frames", 1.0)

    return enhanced_count


def load_model(model_name, callback=None):
    """
    Load super-resolution model

    Parameters:
        model_name: Model name
        callback: Callback function for progress updates

    Returns:
        upsampler: RealESRGANer instance
    """
    if callback:
        callback(f"Loading model: {model_name}")

    model, netscale, _, dni_weight = get_model(model_name)

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(ROOT_DIR, "weights")

    # Find weight file
    model_path = None
    if model_name == "RealESRGAN_x4plus":
        model_path = os.path.join(weights_dir, "RealESRGAN_x4plus.pth")
    elif model_name == "RealESRNet_x4plus":
        model_path = os.path.join(weights_dir, "RealESRNet_x4plus.pth")
    elif model_name == "RealESRGAN_x4plus_anime_6B":
        model_path = os.path.join(weights_dir, "RealESRGAN_x4plus_anime_6B.pth")
    elif model_name == "RealESRGAN_x2plus":
        model_path = os.path.join(weights_dir, "RealESRGAN_x2plus.pth")
    elif model_name == "realesr-animevideov3":
        model_path = os.path.join(weights_dir, "realesr-animevideov3.pth")
    elif model_name == "realesr-general-x4v3":
        model_path = os.path.join(weights_dir, "realesr-general-x4v3.pth")

    if not os.path.exists(model_path):
        if callback:
            callback(f"Error: Model weight file not found {model_path}")
        raise FileNotFoundError(f"Model weight file not found {model_path}")

    if callback:
        callback(f"Initializing super-resolution model...")

    # upsampler = RealESRGANer(
    #     scale=netscale,
    #     model_path=model_path,
    #     dni_weight=dni_weight,
    #     model=model,
    #     tile=0,
    #     tile_pad=10,
    #     pre_pad=0,
    #     half=False,
    #     gpu_id=-1,
    #     device="cpu",
    # )
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=0,
        device="cuda",
    )
    if callback:
        callback(f"Model loading complete: {model_name}")

    return upsampler


def process_video(video_path, output_dir=None, params=None, callback=None):
    """
    Process video: Extract frames and optionally enhance

    Parameters:
        video_path: Video file path
        output_dir: Output directory, creates random directory if None
        params: Processing parameter dictionary, containing:
            - frame_skip: Frame interval
            - enhance: Whether to perform super-resolution
            - model_name: Super-resolution model name
            - outscale: Output scale factor
        callback: Callback function for progress updates

    Returns:
        result: Dictionary containing processing results
    """
    # Set default parameters
    if params is None:
        params = {}

    frame_skip = params.get("frame_skip", 5)
    enhance = params.get("enhance", True)
    model_name = params.get("model_name", "RealESRGAN_x4plus")
    outscale = params.get("outscale", 2)

    # Create unique working directory
    if output_dir is None:
        output_dir = f"process_{uuid.uuid4().hex[:8]}"

    # Create subdirectories
    frames_dir = os.path.join(output_dir, "frames")
    enhanced_dir = os.path.join(output_dir, "enhanced")

    # Create working directory
    create_directory(output_dir)

    if callback:
        callback(f"Creating working directory: {output_dir}")
        callback(
            f"Processing parameters: frame_skip={frame_skip}, enhance={enhance}, model={model_name}, scale={outscale}"
        )

    # Step 1: Extract frames
    frames_count = extract_frames(
        video_path=video_path,
        output_dir=frames_dir,
        frame_skip=frame_skip,
        callback=callback,
    )

    result = {"frames_dir": frames_dir, "frames_count": frames_count, "enhanced": False}

    # If no frames extracted, return directly
    if frames_count == 0:
        if callback:
            callback("Error: Failed to extract any frames")
        return result

    # Step 2: If enhancement enabled, perform super-resolution processing
    if enhance:
        # Load model
        upsampler = load_model(model_name, callback=callback)

        # Enhance frames
        enhanced_count = enhance_frames(
            input_dir=frames_dir,
            output_dir=enhanced_dir,
            upsampler=upsampler,
            outscale=outscale,
            callback=callback,
        )

        if enhanced_count > 0:
            result["enhanced"] = True
            result["enhanced_dir"] = enhanced_dir
            result["enhanced_count"] = enhanced_count

            if callback:
                callback(
                    f"Super-resolution processing complete! Enhanced {enhanced_count} frames"
                )
        else:
            if callback:
                callback("Warning: Failed to enhance any frames")

    return result


def main():
    video_path = "8290394d23cb3589e70f6060b13c2592.mp4"
    output_dir = f"process_{uuid.uuid4().hex[:8]}"

    # Define callback function
    def print_callback(message, progress=None):
        progress_info = f"({progress*100:.1f}%)" if progress is not None else ""
        print(f"{message} {progress_info}")

    # Set processing parameters
    params = {
        "frame_skip": 5,
        "enhance": True,
        "model_name": "RealESRGAN_x4plus",
        "outscale": 2,
    }

    # Process video
    result = process_video(
        video_path=video_path,
        output_dir=output_dir,
        params=params,
        callback=print_callback,
    )

    print("\nProcessing results:")
    print(f"Frames directory: {result['frames_dir']}")
    print(f"Extracted frames: {result['frames_count']}")

    if result["enhanced"]:
        print(f"Enhanced directory: {result['enhanced_dir']}")
        print(f"Enhanced frames: {result['enhanced_count']}")


if __name__ == "__main__":
    main()
