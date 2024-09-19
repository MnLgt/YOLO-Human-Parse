import os
from functools import lru_cache
from typing import List

import cv2
import numpy as np
from diffusers.utils import load_image
from PIL import Image, ImageChops, ImageFilter
from ultralytics import YOLO
from .utils import *


def dilate_mask(mask, dilate_factor=6, blur_radius=2, erosion_factor=2):
    if not mask:
        return None
    # Convert PIL image to NumPy array if necessary
    if isinstance(mask, Image.Image):
        mask = np.array(mask)

    # Ensure mask is in uint8 format
    mask = mask.astype(np.uint8)

    # Apply dilation
    kernel = np.ones((dilate_factor, dilate_factor), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    # Apply erosion for refinement
    kernel = np.ones((erosion_factor, erosion_factor), np.uint8)
    eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)

    # Apply Gaussian blur to smooth the edges
    blurred_mask = cv2.GaussianBlur(
        eroded_mask, (2 * blur_radius + 1, 2 * blur_radius + 1), 0
    )

    # Convert back to PIL image
    smoothed_mask = Image.fromarray(blurred_mask).convert("L")

    # Optionally, apply an additional blur for extra smoothness using PIL
    smoothed_mask = smoothed_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return smoothed_mask


@lru_cache(maxsize=1)
def get_model(model_id):
    model = YOLO(model=model_id)
    return model


def combine_masks(masks: List[dict], labels: List[str], is_label=True) -> Image.Image:
    """
    Combine masks with the specified labels into a single mask, optimized for speed and non-overlapping of excluded masks.

    Parameters:
    - masks (List[dict]): A list of dictionaries, each containing the mask under a 'mask' key and its label under a 'label' key.
    - labels (List[str]): A list of labels to include in the combination.

    Returns:
    - Image.Image: The combined mask as a PIL Image object, or None if no masks are combined.
    """
    labels_set = set(labels)  # Convert labels list to a set for O(1) lookups

    # Filter out any masks that do not have a label key
    masks = [mask for mask in masks if "label" in mask]

    # Filter and convert mask images based on the specified labels
    mask_images = [
        mask["mask"].convert("L")
        for mask in masks
        if (mask["label"] in labels_set) == is_label
    ]

    # Ensure there is at least one mask to combine
    if not mask_images:
        return None  # Or raise an appropriate error, e.g., ValueError("No masks found for the specified labels.")

    # Initialize the combined mask with the first mask
    combined_mask = mask_images[0]

    # Combine the remaining masks with the existing combined_mask using a bitwise OR operation to ensure non-overlap
    for mask in mask_images[1:]:
        combined_mask = ImageChops.lighter(combined_mask, mask)

    return combined_mask


body_labels = ["hair", "face", "arm", "hand", "leg", "foot", "outfit"]


class BodyMask:
    def __init__(
        self,
        image_path,
        model_id,
        labels=body_labels,
        overlay="mask",
        widen_box=0,
        elongate_box=0,
        resize_to=640,
        dilate_factor=0,
        is_label=False,
        resize_to_nearest_eight=False,
        verbose=True,
        remove_overlap=True,
    ):
        self.image_path = image_path
        self.image = self.get_image(
            resize_to=resize_to, resize_to_nearest_eight=resize_to_nearest_eight
        )
        self.labels = labels
        self.is_label = is_label
        self.model_id = model_id
        self.model = get_model(self.model_id)
        self.model_labels = self.model.names
        self.verbose = verbose
        self.results = self.get_results()
        self.dilate_factor = dilate_factor
        self.body_mask = self.get_body_mask()
        self.box = self.get_bounding_box()
        self.body_box = self.get_body_box(
            remove_overlap=remove_overlap, widen=widen_box, elongate=elongate_box
        )
        self.overlay = self.create_overlay(overlay)

    def get_image(self, resize_to, resize_to_nearest_eight):
        image = load_image(self.image_path)
        if resize_to:
            image = resize_preserve_aspect_ratio(image, resize_to)
        if resize_to_nearest_eight:
            image = resize_image_to_nearest_eight(image)
        return image

    def get_results(self):
        imgsz = max(self.image.size)
        results = self.model(
            self.image, retina_masks=True, imgsz=imgsz, verbose=self.verbose
        )[0]
        masks, boxes, scores, phrases = unload(results, self.model_labels)
        results = format_results(
            masks, boxes, scores, phrases, self.model_labels, person_masks_only=False
        )
        masks_to_filter = ["hair"]
        results = filter_highest_score(results, ["hair", "face", "phone"])
        return results

    def get_body_mask(self):
        body_mask = combine_masks(self.results, self.labels, self.is_label)
        if body_mask is not None:
            return dilate_mask(body_mask, self.dilate_factor)
        return None

    def get_bounding_box(self):
        if self.body_mask is None:
            return None
        return get_bounding_box(self.body_mask)

    def get_body_box(self, remove_overlap=True, widen=0, elongate=0):
        if self.body_mask is None or self.box is None:
            return None
        body_box = get_bounding_box_mask(self.body_mask, widen=widen, elongate=elongate)
        if remove_overlap and body_box is not None:
            body_box = self.remove_overlap(body_box)
        return body_box

    def create_overlay(self, overlay_type):
        if self.body_box is not None and overlay_type == "box":
            return overlay_mask(self.image, self.body_box, opacity=0.9, color="red")
        elif self.body_mask is not None:
            return overlay_mask(self.image, self.body_mask, opacity=0.9, color="red")
        return self.image

    def remove_overlap(self, body_box):
        if body_box is None:
            return None
        box_array = np.array(body_box)
        mask = self.combine_masks(mask_labels=self.labels, is_label=True)
        if mask is None:
            return body_box
        mask_array = np.array(mask)
        box_array[mask_array == 255] = 0
        return Image.fromarray(box_array)

    def combine_masks(self, mask_labels: List, no_labels=None, is_label=True):
        if not is_label:
            mask_labels = [
                phrase for phrase in self.phrases if phrase not in mask_labels
            ]
        masks = [
            row.get("mask") for row in self.results if row.get("label") in mask_labels
        ]
        if len(masks) == 0:
            return None
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = ImageChops.lighter(combined_mask, mask)
        return combined_mask

    def display_results(self):
        if not self.results:
            print("No results to display.")
            return
        cols = min(len(self.results), 4)
        display_image_with_masks(self.image, self.results, cols=cols)

    def get_mask(self, mask_label):
        if mask_label not in self.phrases:
            print(f"Mask label '{mask_label}' not found in results.")
            return None
        return [f for f in self.results if f.get("label") == mask_label]
