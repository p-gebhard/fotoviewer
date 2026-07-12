from fotoviewer.image_processor.indexer import Indexer
from fotoviewer.image_processor.image_processor import ImageProcessor


if __name__ == "__main__":
    ip = ImageProcessor()
    indexer = Indexer(ip.compute_image_features)
    indexer.index_folder("/home/user/Pictures/personal/", "../indices")
