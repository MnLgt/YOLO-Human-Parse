from typing import List, Union
from PIL import Image
from ultralytics import YOLO

from hp.visualizer import visualizer
from .utils import *


class YOLOResults:
    def __init__(self, image: Union[Image.Image | str], result: List):
        self.image = image
        self.masks = None
        self.boxes = None
        self.scores = None
        self.labels = None
        self.labels_dict = None
        self.result = self.unload(result[0])
        self.formatted_results = format_results(
            self.labels,
            self.scores,
            self.boxes,
            self.masks,
        )

    def unload(self, result):
        assert (
            bool(result) and hasattr(result, "masks") and hasattr(result, "boxes")
        ), "No Masks or Boxes Found"
        self.masks = unload_masks(result.masks.data)
        self.boxes = unload_boxes(result.boxes.xyxy)
        self.scores = format_scores(result.boxes.conf)
        self.labels = list(result.names.values())
        self.labels_dict = result.names
        det_ids = result.boxes.cls
        det_ids = [int(l.item()) for l in det_ids]
        self.labels = [self.labels_dict[i] for i in det_ids]

    def visualize(self, return_image=False):
        return visualizer(
            self.image,
            self.formatted_results,
            prompt_label="label",
            return_image=return_image,
        )
