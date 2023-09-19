import os
import numpy as np
import cv2
import torch
from PIL import Image
import time

from segment_anything import sam_model_registry, SamPredictor

def sam_init(device_id=0):
    sam_checkpoint = os.path.join(os.path.dirname(__file__), "../../sam_vit_h_4b8939.pth")
    model_type = "vit_h"

    device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def sam_out_nosave(predictor, input_image, *bbox_sliders):
    bbox = np.array(bbox_sliders)
    image = np.asarray(input_image)

    start_time = time.time()
    predictor.set_image(image)

    h, w, _ = image.shape
    input_point = np.array([[h//2, w//2]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(
        box=bbox,
        multimask_output=True
    )

    print(f"SAM Time: {time.time() - start_time:.3f}s")
    opt_idx = np.argmax(scores)
    mask = masks[opt_idx]
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image[:, :, 3] = mask.astype(np.uint8) * 255
    out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255 # np.argmax(scores_bbox)
    torch.cuda.empty_cache()
    return Image.fromarray(out_image_bbox, mode='RGBA') 

def image_preprocess_nosave(input_image, lower_contrast=True, rescale=True):

    image_arr = np.array(input_image)
    in_w, in_h = image_arr.shape[:2]

    if lower_contrast:
        alpha = 0.8  # Contrast control (1.0-3.0)
        beta =  0   # Brightness control (0-100)
        # Apply the contrast adjustment
        image_arr = cv2.convertScaleAbs(image_arr, alpha=alpha, beta=beta)
        image_arr[image_arr[...,-1]>200, -1] = 255

    ret, mask = cv2.threshold(np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    ratio = 0.75
    if rescale:
        side_len = int(max_size / ratio)
    else:
        side_len = in_w
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len//2
    padded_image[center-h//2:center-h//2+h, center-w//2:center-w//2+w] = image_arr[y:y+h, x:x+w]
    rgba = Image.fromarray(padded_image).resize((256, 256), Image.LANCZOS)

    rgba_arr = np.array(rgba) / 255.0
    rgb = rgba_arr[...,:3] * rgba_arr[...,-1:] + (1 - rgba_arr[...,-1:])
    return Image.fromarray((rgb * 255).astype(np.uint8))