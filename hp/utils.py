import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def unload_mask(mask):
    mask = mask.cpu().numpy().squeeze()
    mask = mask.astype(np.uint8) * 255
    return Image.fromarray(mask)


def unload_box(box):
    return box.cpu().numpy().tolist()


def masks_overlap(mask1, mask2):
    return np.any(np.logical_and(mask1, mask2))


def remove_non_person_masks(person_mask, formatted_results):
    return [
        f
        for f in formatted_results
        if f.get("label") == "person" or masks_overlap(person_mask, f.get("mask"))
    ]


def format_masks(masks):
    return [unload_mask(mask) for mask in masks]


def format_boxes(boxes):
    return [unload_box(box) for box in boxes]


def format_scores(scores):
    return scores.cpu().numpy().tolist()


def unload(result, labels_dict):
    masks = format_masks(result.masks.data)
    boxes = format_boxes(result.boxes.xyxy)
    scores = format_scores(result.boxes.conf)
    labels = result.boxes.cls
    labels = [int(label.item()) for label in labels]
    phrases = [labels_dict[label] for label in labels]
    return masks, boxes, scores, phrases


def format_results(masks, boxes, scores, labels, labels_dict, person_masks_only=True):
    if isinstance(list(labels_dict.keys())[0], int):
        labels_dict = {v: k for k, v in labels_dict.items()}

    # check that the person mask is present
    if person_masks_only:
        assert "person" in labels, "Person mask not present in results"
    results_dict = []
    for row in zip(labels, scores, boxes, masks):
        label, score, box, mask = row
        label_id = labels_dict[label]
        results_row = dict(
            label=label, score=score, mask=mask, box=box, label_id=label_id
        )
        results_dict.append(results_row)
    results_dict = sorted(results_dict, key=lambda x: x["label"])
    if person_masks_only:
        # Get the person mask
        person_mask = [f for f in results_dict if f.get("label") == "person"][0]["mask"]
        assert person_mask is not None, "Person mask not found in results"

        # Remove any results that do no overlap with the person
        results_dict = remove_non_person_masks(person_mask, results_dict)
    return results_dict


def filter_highest_score(results, labels):
    """
    Filter results to remove entries with lower scores for specified labels.

    Args:
        results (list): List of dictionaries containing 'label', 'score', and other keys.
        labels (list): List of labels to filter.

    Returns:
        list: Filtered results with only the highest score for each specified label.
    """
    # Dictionary to keep track of the highest score entry for each label
    label_highest = {}

    # First pass: identify the highest score for each label
    for result in results:
        label = result["label"]
        if label in labels:
            if (
                label not in label_highest
                or result["score"] > label_highest[label]["score"]
            ):
                label_highest[label] = result

    # Second pass: construct the filtered list while preserving the order
    filtered_results = []
    seen_labels = set()

    for result in results:
        label = result["label"]
        if label in labels:
            if label in seen_labels:
                continue
            if result == label_highest[label]:
                filtered_results.append(result)
                seen_labels.add(label)
        else:
            filtered_results.append(result)

    return filtered_results


def display_image_with_masks(image, results, cols=4, return_images=False):
    # Convert PIL Image to numpy array
    image_np = np.array(image)

    # Check image dimensions
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("Image must be a 3-dimensional array with 3 color channels")

    # Number of masks
    n = len(results)
    rows = (n + cols - 1) // cols  # Calculate required number of rows

    # Setting up the plot
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axs = np.array(axs).reshape(-1)  # Flatten axs array for easy indexing
    for i, result in enumerate(results):
        mask = result["mask"]
        label = result["label"]
        score = float(result["score"])

        # Convert PIL mask to numpy array and resize if necessary
        mask_np = np.array(mask)
        if mask_np.shape != image_np.shape[:2]:
            mask_np = resize(
                mask_np, image_np.shape[:2], mode="constant", anti_aliasing=False
            )
            mask_np = (mask_np > 0.5).astype(
                np.uint8
            )  # Threshold back to binary after resize

        # Create an overlay where mask is True
        overlay = np.zeros_like(image_np)
        overlay[mask_np > 0] = [0, 0, 255]  # Applying blue color on the mask area

        # Combine the image and the overlay
        combined = image_np.copy()
        indices = np.where(mask_np > 0)
        combined[indices] = combined[indices] * 0.5 + overlay[indices] * 0.5

        # Show the combined image
        ax = axs[i]
        ax.imshow(combined)
        ax.axis("off")
        ax.set_title(f"Label: {label}, Score: {score:.2f}", fontsize=12)
        rect = patches.Rectangle(
            (0, 0),
            image_np.shape[1],
            image_np.shape[0],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    # Hide unused subplots if the total number of masks is not a multiple of cols
    for idx in range(i + 1, rows * cols):
        axs[idx].axis("off")
    plt.tight_layout()
    plt.show()


def get_bounding_box(mask):
    """
    Given a segmentation mask, return the bounding box for the mask object.
    """
    # Find indices where the mask is non-zero
    coords = np.argwhere(mask)
    # Get the minimum and maximum x and y coordinates
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    # Return the bounding box coordinates
    return (y_min, x_min, y_max, x_max)


def get_bounding_box_mask(segmentation_mask, widen=0, elongate=0):
    # Convert the PIL segmentation mask to a NumPy array
    mask_array = np.array(segmentation_mask)

    # Find the coordinates of the non-zero pixels
    non_zero_y, non_zero_x = np.nonzero(mask_array)

    # Calculate the bounding box coordinates
    min_x, max_x = np.min(non_zero_x), np.max(non_zero_x)
    min_y, max_y = np.min(non_zero_y), np.max(non_zero_y)

    if widen > 0:
        min_x = max(0, min_x - widen)
        max_x = min(mask_array.shape[1], max_x + widen)

    if elongate > 0:
        min_y = max(0, min_y - elongate)
        max_y = min(mask_array.shape[0], max_y + elongate)

    # Create a new blank image for the bounding box mask
    bounding_box_mask = Image.new("1", segmentation_mask.size)

    # Draw the filled bounding box on the blank image
    draw = ImageDraw.Draw(bounding_box_mask)
    draw.rectangle([(min_x, min_y), (max_x, max_y)], fill=1)

    return bounding_box_mask


colors = {
    "blue": (136, 207, 249),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "lime": (50, 205, 50),
    "pink": (255, 192, 203),
    "brown": (139, 69, 19),
    "gray": (128, 128, 128),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gold": (255, 215, 0),
    "silver": (192, 192, 192),
    "beige": (245, 245, 220),
    "navy": (0, 0, 128),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
}


def overlay_mask(image, mask, opacity=0.5, color="blue"):
    """
    Takes in a PIL image and a PIL boolean image mask. Overlay the mask on the image
    and color the mask with a low opacity blue with hex #88CFF9.
    """
    # Convert the boolean mask to an image with alpha channel
    alpha = mask.convert("L").point(lambda x: 255 if x == 255 else 0, mode="1")

    # Choose the color
    r, g, b = colors[color]

    color_mask = Image.new("RGBA", mask.size, (r, g, b, int(opacity * 255)))
    mask_rgba = Image.composite(
        color_mask, Image.new("RGBA", mask.size, (0, 0, 0, 0)), alpha
    )

    # Create a new RGBA image to overlay the mask on
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

    # Paste the mask onto the overlay
    overlay.paste(mask_rgba, (0, 0))

    # Create a new image to return by blending the original image and the overlay
    result = Image.alpha_composite(image.convert("RGBA"), overlay)

    # Convert the result back to the original mode and return it
    return result.convert(image.mode)


def resize_preserve_aspect_ratio(image, max_side=512):
    width, height = image.size
    scale = min(max_side / width, max_side / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return image.resize((new_width, new_height))


def round_to_nearest_eigth(value):
    return int((value // 8 * 8))


def resize_image_to_nearest_eight(image):
    width, height = image.size
    width, height = round_to_nearest_eigth(width), round_to_nearest_eigth(height)
    image = image.resize((width, height))
    return image
