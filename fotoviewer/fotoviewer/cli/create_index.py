import argparse

from fotoviewer.core.config import Settings
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.indexer import Indexer
from fotoviewer.services.reindex import image_model_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the search index (only new/modified photos).")
    parser.add_argument("--full", action="store_true", help="rebuild the index from scratch")
    args = parser.parse_args()

    settings = Settings()
    ip = ImageProcessor(settings.embedding_model_name)
    indexer = Indexer(ip.compute_image_features, image_model_key(ip))
    indexer.index_folder(settings.images_path, settings.index_dir, full=args.full)


if __name__ == "__main__":
    main()
