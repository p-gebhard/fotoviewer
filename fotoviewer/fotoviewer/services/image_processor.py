from typing import Any
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import normalize


class ImageProcessor:

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self._model = CLIPModel.from_pretrained(model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name)

    def compute_image_features(self, image: Any) -> np.ndarray:
        inputs = self._processor(images=image, return_tensors="pt")
        image_embeds = self._model.get_image_features(**inputs).pooler_output
        return normalize(image_embeds.detach()).cpu().numpy()

    def compute_text_features(self, text: str) -> np.ndarray:
        inputs = self._processor(text=text, return_tensors="pt", padding=True)
        text_embeds = self._model.get_text_features(**inputs).pooler_output
        return normalize(text_embeds.detach()).cpu().numpy()


if __name__ == "__main__":
    img_file = "/home/user/projects/fotoviewer/images/CIMG2568.JPG"
    image = Image.open(img_file)
    ip = ImageProcessor()
    print(ip.compute_image_features(image))
    print(ip.compute_text_features("a woman sitting on a camel"))
