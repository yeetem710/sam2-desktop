import cv2
import numpy as np
from PIL import ImageGrab
import time
import os

def countdown(seconds):
    for i in range(seconds, 0, -1):
        print(f"Starting in {i} seconds...")
        time.sleep(1)

def capture_and_process_frames(duration=10, fps=30):
    # Create a directory to store frames
    os.makedirs("input-dir", exist_ok=True)

    # Calculate the time interval between frames
    frame_interval = 1 / fps

    input("Press Enter to start the 15-second countdown...")
    
    countdown(15)

    print("Capturing started!")
    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < duration:
        # Capture the screen
        screenshot = ImageGrab.grab()

        # Convert to a numpy array
        frame = np.array(screenshot)

        # Convert from RGB to BGR (OpenCV uses BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Save the frame as JPEG
        cv2.imwrite(f"input-data/frame_{frame_count:04d}.jpeg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Process the frame (example: apply a Gaussian blur)
        # processed_frame = cv2.GaussianBlur(frame, (15, 15), 0)

        # Save the processed frame as JPEG
       #  cv2.imwrite(f"input-data/processed_frame_{frame_count:04d}.jpeg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        frame_count += 1

        # Wait for the next frame interval
        time.sleep(max(0, start_time + frame_count * frame_interval - time.time()))

    print(f"Captured and processed {frame_count} frames in {duration} seconds")

if __name__ == "__main__":
    capture_and_process_frames(duration=2, fps=10)