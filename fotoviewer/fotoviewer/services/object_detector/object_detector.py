import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


class ObjectDetector:

    def __init__(self, device, model_name: str = "PekingU/rtdetr_r50vd") -> None:
        self._device = device
        self._model = RTDetrForObjectDetection.from_pretrained(model_name).to(device)
        self._model.eval()
        self._processor = RTDetrImageProcessor.from_pretrained(model_name)

    def detect(self, im):
        inputs = self._processor(images=im, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([im.size[::-1]])
        results = self._processor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=target_sizes
        )[0]

        return [
            {"label": self._model.config.id2label[label_id.item()], "score": score.item(), "bbox": box.tolist()}
            for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"])
        ]


if __name__ == "__main__":
    from PIL import Image

    from fotoviewer.services.object_detector.visualizer import render_bboxes

    img_file = "/home/user/projects/fotoviewer/images/CIMG2568.JPG"
    image = Image.open(img_file)
    od = ObjectDetector("cuda")
    detections = od.detect(image)
    print(detections)
    render_bboxes(image, detections)
    image.save(img_file.replace(".JPG", "_boxes.JPG"))
