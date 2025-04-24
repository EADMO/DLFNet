import os
import cv2
import re

def extract_number(filename):
    """Extract the numeric part from filename for sorting"""
    # Match the number at the end of filename (e.g., 04380 from _04380.jpg)
    match = re.search(r'_(\d+)\.jpg$', filename)
    if match:
        return int(match.group(1))
    return 0  # Return 0 if no number is found

def create_video_from_images(image_folder, output_video, prefix, fps=30):
    """Create video from sequence of images"""
    
    # Get all jpg files matching the prefix
    image_files = [f for f in os.listdir(image_folder) 
                  if f.startswith(prefix) and f.endswith('.jpg')]
    
    if not image_files:
        print(f"Error: No jpg files found with prefix '{prefix}' in {image_folder}")
        return
    
    # Sort files using extracted numbers
    image_files.sort(key=extract_number)
    
    # Read first image to get video dimensions
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    if first_image is None:
        print(f"Error: Failed to read first image {image_files[0]}")
        return
    height, width, _ = first_image.shape
    
    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Write frames to video
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Failed to read image {image_file}, skipping")
            continue
        video.write(frame)
        print(f"Added frame: {image_file}")
    
    # Release video writer
    video.release()
    print(f"Video successfully created: {output_video}")

# Configuration
image_folder = "work_dirs/dlf/dla34_culane/20250424_203613_lr_6e-04_b_24/visualization" # Replace with your visualization path
# prefix = "driver_100_30frame_05252249_0542.MP4_"
prefix = "driver_100_30frame_05251204_0379.MP4_"
output_video = "output_video.mp4"
fps = 10  # Frame rate, adjustable as needed

# Execute video creation
create_video_from_images(image_folder, output_video, prefix, fps)