from PIL import Image, ImageDraw


def render_bboxes(image: Image, bboxes):
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle(bbox)
    draw = ImageDraw.Draw(image)
