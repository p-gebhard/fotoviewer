from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

from fotoviewer.services.object_detector.visualizer import render_bboxes
from fotoviewer.services.object_detector.detr_model import DETRModel


DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent / "detr_demo-da2a99e9.pth"


class ObjectDetector():

    def __init__(self, device, checkpoint_path: str | Path | None = None) -> None:
        checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
        self._model = DETRModel(num_classes=91)
        state_dict = torch.load(checkpoint_path, map_location=device)
        self._model.load_state_dict(state_dict)
        self._model.eval()

        self._transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def rescale_bboxes(self, out_bbox, size):
        def box_cxcywh_to_xyxy(x):
            x_c, y_c, w, h = x.unbind(1)
            b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                (x_c + 0.5 * w), (y_c + 0.5 * h)]
            return torch.stack(b, dim=1)

        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def detect(self, im):
        img = self._transform(im).unsqueeze(0)
        assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, "demo model only supports images up to 1600 pixels on each side"

        outputs = self._model(img)

        probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.7

        bboxes_scaled = self.rescale_bboxes(outputs["pred_boxes"][0, keep], im.size)
        return probas[keep].tolist(), bboxes_scaled.tolist()


if __name__ == "__main__":
    img_file = "/home/user/projects/fotoviewer/images/CIMG2568.JPG"
    image = Image.open(img_file)
    od = ObjectDetector("cuda")
    scores, boxes = od.detect(image)
    print(scores)
    print(boxes)
    render_bboxes(image, boxes)
    image.save(img_file.replace(".JPG", "_boxes.JPG"))
