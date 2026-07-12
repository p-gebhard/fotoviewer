from PIL import Image

from fotoviewer.image_processor.index import Index
from fotoviewer.image_processor.image_processor import ImageProcessor


if __name__ == "__main__":
    ip = ImageProcessor()
    index = Index()
    index.load("image", "../indices/image.npz")
    img = Image.open("../images/IMG_1523.JPG")
    img_features = ip.compute_image_features(img)
    print(index.search("image", img_features, 10))
