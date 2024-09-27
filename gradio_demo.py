import gradio as gr
import os
from hp.yolo_results import YOLOResults
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
import io
from functools import lru_cache
import logging
from ultralytics import YOLO
from hp.utils import load_resize_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_id = os.path.abspath("weights/yolo-human-parse-v2.pt")


@lru_cache
def get_model(model_id=model_id):
    return YOLO(model_id, task="segment")


def perform_segmentation(image):
    model = get_model()
    image = load_resize_image(image, 1024)
    imgsz = max(image.size)
    result = model(image, imgsz=imgsz, retina_masks=True)
    if not bool(result):
        logger.info("No Masks or Boxes Found")
        return image
    result = YOLOResults(image, result)
    return result.visualize(return_image=True)


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
