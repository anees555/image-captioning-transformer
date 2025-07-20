from PIL import Image
from src.dataset import my_transforms

def preprocess_image(image_path):
    """
    Loads and transforms a single image for inference.
    """
    image = Image.open(image_path).convert("RGB")
    image = my_transforms(image)
    return image
