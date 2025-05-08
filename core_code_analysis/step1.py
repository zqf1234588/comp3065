import cv2
import os
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url
import glob



# Create directory to store frames
if not os.path.exists("frames"):
    os.makedirs("frames")

# Open the video
video = cv2.VideoCapture("8290394d23cb3589e70f6060b13c2592.mp4")

# Check if video opened successfully
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Variables
count = 0
frame_skip = 1  # Extract every 5th frame (adjust based on video length and motion)

while True:
    # Read next frame
    success, frame = video.read()
    
    # Break if no more frames
    if not success:
        break
    
    # Save frame only every 'frame_skip' frames
    if count % frame_skip == 0:
        cv2.imwrite(f"frames/frame_{count:04d}.jpg", frame)
        print(f"Saved frame {count}")
    
    count += 1

# Release the video
video.release()
print(f"Total frames extracted: {count//frame_skip}")



output_dir = "enhanced_frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model_name = 'RealESRGAN_x4plus'  # options: RealESRGAN_x4plus | RealESRGAN_x4plus_anime_6B

if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    dni_weight = None
elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    netscale = 2
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    netscale = 4
    file_url = [
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
    ]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
for url in file_url:
    # model_path will be updated
    model_path = load_file_from_url(
        url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

upsampler = RealESRGANer(
    scale=netscale,
    model_path=model_path,
    dni_weight=dni_weight,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True,  # use half precision during inference
    gpu_id=0    # set to None for CPU mode
)


# Get all frames
frame_paths = sorted(glob.glob('frames/frame_*.jpg'))
total_frames = len(frame_paths)

# Process each frame
for idx, path in enumerate(frame_paths):
    # Read image
    frame_number = os.path.basename(path).split('_')[1].split('.')[0]
    output_path = os.path.join(output_dir, f"enhanced_{frame_number}.png")
    
    # Read image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    # Enhance image (super-resolution + deblurring)
    output, _ = upsampler.enhance(img, outscale=4)
    
    # Save enhanced image
    cv2.imwrite(output_path, output)
    
    print(f"Enhanced frame {idx+1}/{total_frames}: {path} -> {output_path}")