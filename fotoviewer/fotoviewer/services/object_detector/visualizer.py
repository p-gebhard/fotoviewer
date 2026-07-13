from PIL import Image, ImageDraw


def render_bboxes(image: Image, detections):
    draw = ImageDraw.Draw(image)
    for detection in detections:
        draw.rectangle(detection["bbox"])
        draw.text((detection["bbox"][0], detection["bbox"][1]), detection["label"])
