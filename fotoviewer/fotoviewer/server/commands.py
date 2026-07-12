import base64
import torch
from io import BytesIO
from PIL import Image

from fotoviewer.object_detector.object_detector import ObjectDetector
from fotoviewer.object_detector.visualizer import render_bboxes
from fotoviewer.image_processor.image_processor import ImageProcessor
from fotoviewer.image_processor.index import Index


device = "cuda" if torch.cuda.is_available() else "cpu"
object_detector = ObjectDetector(device)

ip = ImageProcessor()
index = Index()
index.load("image", "../indices/image.npz")


def detect_objects_in_image(params):
    img = base64.b64decode(params["image"])
    image = Image.open(BytesIO(img))
    scores, boxes = object_detector.detect(image)
    render_bboxes(image, boxes)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return {
        "image": img_str.decode("utf-8"),
        "bboxes": boxes,
        "scores": scores
    }


def search_similar_images(params):
    img = base64.b64decode(params["image"])
    image = Image.open(BytesIO(img))
    img_features = ip.compute_image_features(image)
    return index.search("image", img_features, 100)


def search_images_by_text(params):
    query = params["query"]
    text_features = ip.compute_text_features(query)
    return index.search("image", text_features, 100)
