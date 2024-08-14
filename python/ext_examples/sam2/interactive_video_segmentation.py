import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from sam2.build_sam import build_sam2_video_predictor
from tempfile import TemporaryDirectory
import os

def capture_and_select_pixel(image):
    coords = []

    def select_pixel(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Pixel selected at (X: {x}, Y: {y}) - Color: {param[y, x]}")
            coords.append((x, y))

    cv2.imshow("Captured Image", image)
    cv2.setMouseCallback("Captured Image", select_pixel, image)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    if coords:
        return coords[0]
    else:
        return None

def extract_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"{frame_index}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Extracted frame {frame_index} to {frame_filename}")
        frame_index += 1
    cap.release()
    print(f"Extracted {frame_index} frames to {output_dir}")

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

with TemporaryDirectory() as temp_dir:
    extract_frames("images.mp4", temp_dir)

    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        state = predictor.init_state(temp_dir)
        file_frame_path = os.path.join(temp_dir, "0.jpg")
        image = cv2.imread(file_frame_path)
        coords = np.array([capture_and_select_pixel(image)])
        frame_idx, object_ids, masks = predictor.add_new_points_or_box(
            state, 0, 1, coords, [1])
        masks_list = []
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            masks_np = masks[0][0].cpu().detach().numpy() 
            masks_list.append(masks_np)
            plt.imshow(masks_np > 0)
            plt.show()
