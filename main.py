import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

def process_image(image_path, predictor):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    predictor.set_image(image)
    
    # Generate automatic masks
    masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=None)
    
    # Visualize the masks
    for i, mask in enumerate(masks):
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colored_mask[mask] = np.random.randint(0, 255, 3)
        
        overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        cv2.imwrite(f"output_{os.path.basename(image_path)}_{i}.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def process_video(video_path, predictor):
    cap = cv2.VideoCapture(video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = f"output_{os.path.basename(video_path)}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(frame_rgb)
        
        masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=None)
        
        colored_mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        for mask in masks:
            colored_mask[mask] = np.random.randint(0, 255, 3)
        
        overlay = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
        out.write(overlay)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    
    cap.release()
    out.release()

def main():
    # Load the SAM model
    sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"  # Make sure this file is in the same directory
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    predictor = SamPredictor(sam)
    
    input_dir = "input-data"
    
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {filename}")
            process_image(file_path, predictor)
        elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
            print(f"Processing video: {filename}")
            process_video(file_path, predictor)
        else:
            print(f"Skipping unsupported file: {filename}")

if __name__ == "__main__":
    main()