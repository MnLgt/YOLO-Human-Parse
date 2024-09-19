import gradio as gr
import os
from ultralytics import YOLO
from hp.BodyMask import BodyMask
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage.transform import resize
from PIL import Image
import io

model_id = os.path.abspath("weights/yolo-human-parse-epoch-125.pt")


def display_image_with_masks(image, results, cols=4):
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

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Clear the current figure
    plt.close(fig)

    return buf


def perform_segmentation(input_image):
    bm = BodyMask(input_image, model_id=model_id, resize_to=640)
    if bm.body_mask is None:
        return input_image  # Return the original image if no mask is found
    results = bm.results
    buf = display_image_with_masks(input_image, results)

    # Convert BytesIO to PIL Image
    img = Image.open(buf)
    return img


# Get example images
example_images = [
    os.path.join("sample_images", f)
    for f in os.listdir("sample_images")
    if f.endswith((".png", ".jpg", ".jpeg"))
]
# body_labels = ["hair", "face", "arm", "hand", "leg", "foot", "outfit"]

with gr.Blocks() as demo:
    gr.Markdown("# YOLO Human Parse")
    gr.Markdown(
        "Upload an image of a person or select an example to see the YOLO segmentation results."
    )
    gr.Markdown("Labels: hair, face, arm, hand, leg, foot, outfit")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image", height=512)
            segment_button = gr.Button("Perform Segmentation")

        output_image = gr.Image(label="Segmentation Result")

    gr.Examples(
        examples=example_images,
        inputs=input_image,
        outputs=output_image,
        fn=perform_segmentation,
        cache_examples=True,
    )

    segment_button.click(
        fn=perform_segmentation,
        inputs=input_image,
        outputs=output_image,
    )

demo.launch()
