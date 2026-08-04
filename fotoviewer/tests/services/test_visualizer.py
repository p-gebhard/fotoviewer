from PIL import Image

from fotoviewer.services.object_detector.visualizer import render_bboxes


def test_render_bboxes_draws_onto_the_image():
    image = Image.new("RGB", (50, 50), color="white")
    blank = image.copy()

    render_bboxes(image, [{"label": "dog", "score": 0.9, "bbox": [5, 5, 30, 30]}])

    assert list(image.getdata()) != list(blank.getdata())


def test_render_bboxes_prefers_identity_label_over_object_label():
    image = Image.new("RGB", (50, 50), color="white")
    without_identity = image.copy()
    render_bboxes(without_identity, [{"label": "dog", "score": 0.9, "bbox": [5, 5, 30, 30]}])

    with_identity = image.copy()
    render_bboxes(with_identity, [{"label": "dog", "score": 0.9, "bbox": [5, 5, 30, 30], "identity": "Rex"}])

    assert list(with_identity.getdata()) != list(without_identity.getdata())
