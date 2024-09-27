import os
import random
from typing import List, Union

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

from hp.visualizer import visualizer


def resize_image_pil(image_pil, max_size=1024):
    # Ensure image is in RGB
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")

    # Calculate new dimensions preserving aspect ratio
    width, height = image_pil.size
    scale = min(max_size / width, max_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)

    # Calculate padding needed to reach 1024x1024
    pad_width = (max_size - new_width) // 2
    pad_height = (max_size - new_height) // 2

    # Apply padding symmetrically
    image_pil = ImageOps.expand(
        image_pil,
        border=(
            pad_width,
            pad_height,
            max_size - new_width - pad_width,
            max_size - new_height - pad_height,
        ),
        fill=(0, 0, 0),
    )

    return image_pil


def load_resize_image(image_path: str | Image.Image, size: int) -> Image.Image:
    if isinstance(image_path, str):
        image_pil = Image.open(image_path).convert("RGB")
    else:
        image_pil = image_path.convert("RGB")

    image_pil = resize_image_pil(image_pil, size)
    return image_pil


def unload_mask(mask):
    mask = mask.cpu().numpy().squeeze()
    mask = mask.astype(np.uint8) * 255
    return Image.fromarray(mask)


def unload_masks(masks):
    return [unload_mask(mask) for mask in masks]


def unload_box(box):
    return box.cpu().numpy().tolist()


def unload_boxes(boxes):
    return [unload_box(box) for box in boxes]


def format_scores(scores):
    return scores.squeeze().cpu().numpy().tolist()


def format_results(labels, scores, boxes, masks):
    results_dict = []
    for row in zip(labels, scores, boxes, masks):
        label, score, box, mask = row
        results_row = dict(label=label, score=score, mask=mask, box=box)
        results_dict.append(results_row)
    return results_dict
