# Video-transition-smoothness-using-optical-flow-farneback-approach-
First, it opens original time lapse video and read first frame in video Then it Convert frames to grayscale It estimates pixel motion using  opticalflowfarneback  Apply addWeighted to warp previous frame using flow Blend frame After Save blended frames using VideoWriter Save the smooth transition video   


import os
import cv2
import numpy as np

# Function to compute optical flow between two frames
def compute_optical_flow(prev_gray, next_gray):
    """Compute the optical flow using Farneback method"""
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

# Function to apply optical flow to a color frame
def apply_optical_flow(prev_frame, flow):
    """Apply optical flow to each channel of a color frame"""
    h, w = flow.shape[:2]

    # Generate coordinate grid
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # Compute new pixel locations
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    # Ensure values remain within bounds
    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)

    # Warp each color channel separately
    warped_b = cv2.remap(prev_frame[:, :, 0], map_x, map_y, interpolation=cv2.INTER_LINEAR)
    warped_g = cv2.remap(prev_frame[:, :, 1], map_x, map_y, interpolation=cv2.INTER_LINEAR)
    warped_r = cv2.remap(prev_frame[:, :, 2], map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # Merge channels back into a color frame
    warped_frame = cv2.merge((warped_b, warped_g, warped_r))
    return warped_frame

# Function to blend two frames with optical flow
def blend_frames_with_optical_flow(prev_frame, next_frame, alpha=0.8, beta=0.9):
    """Blend two frames in color using optical flow"""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Compute optical flow
    flow = compute_optical_flow(prev_gray, next_gray)

    # Warp the previous frame using flow
    warped_prev_frame = apply_optical_flow(prev_frame, flow)

    # Blend frames
    blended = cv2.addWeighted(warped_prev_frame, alpha, next_frame, beta, 0)
    return blended

# Function to generate smoothed video using optical flow
def generate_smoothed_video(input_video_path, output_video_path):
    if not os.path.exists(input_video_path):
        print(f"❌ ERROR: The video file ({input_video_path}) does not exist.")
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("❌ ERROR: Could not open video file. Check file format and codecs.")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    ret, prev_frame = cap.read()
    if not ret:
        print("❌ ERROR: Couldn't read the first frame.")
        return

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        # Blend frames using optical flow
        blended = blend_frames_with_optical_flow(prev_frame, next_frame, alpha=0.8, beta=0.9)

        # Write the blended frame to the output video
        out.write(blended)

        prev_frame = next_frame  # Update previous frame

    cap.release()
    out.release()
    print(f"✅ Smoothed color video saved at: {output_video_path}")

# Define the input and output video paths
input_video_path = r"D:\New downloads\Project Work Data\Input\Video.mp4"
output_video_path = r"D:\New downloads\Project Work Data\Output\OutputSmoothedvideo.avi"

# Generate the smoothed video
generate_smoothed_video(input_video_path, output_video_path)
