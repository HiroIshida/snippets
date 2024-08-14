import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2

def capture_and_select_pixel(cap):
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        cap.release()
        return None

    cap.release()
    coords = []

    def select_pixel(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Pixel selected at (X: {x}, Y: {y}) - Color: {param[y, x]}")
            coords.append((x, y))
    cv2.imshow("Captured Image", frame)
    cv2.setMouseCallback("Captured Image", select_pixel, frame)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    if coords:
        return coords[0]
    else:
        return None

if __name__ == "__main__":
    checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    cap = cv2.VideoCapture(4)
    for _ in range(10):
        ret, frame = cap.read()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        ret, frame = cap.read()
        predictor.set_image(frame)
        coords = np.array([capture_and_select_pixel(cap)])
        masks, _, _ = predictor.predict(coords, [1], multimask_output=False)
    plt.imshow(masks[0])
    plt.show()
