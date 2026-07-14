import sys
from pathlib import Path

from PIL import Image

from fotoviewer.core.config import Settings
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.index import Index


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: fotoviewer-search <image-path>")
        raise SystemExit(1)

    settings = Settings()
    ip = ImageProcessor(settings.embedding_model_name)
    index = Index()
    index.load("image", str(Path(settings.index_dir) / "image.npz"))

    image = Image.open(sys.argv[1])
    features = ip.compute_image_features(image)
    for result in index.search("image", features, 10):
        print(result)


if __name__ == "__main__":
    main()
