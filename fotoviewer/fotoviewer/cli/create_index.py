from fotoviewer.core.config import Settings
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.indexer import Indexer


def main() -> None:
    settings = Settings()
    ip = ImageProcessor(settings.clip_model_name)
    indexer = Indexer(ip.compute_image_features)
    indexer.index_folder(settings.images_path, settings.index_dir)


if __name__ == "__main__":
    main()
