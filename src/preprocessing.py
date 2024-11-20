import torch
import torchvision.transforms as T
from PIL import Image

#function to load and preprocess an image
def load_image(img_path, max_size=400):
    """
    Load and preprocess an image from a given path.
    Args:
    - imag_path (str): path to the image file.
    - max_size (int): the maximum size for resizing the image.
    
    Returns:
    - Tensor: a tensor representing the preprocessed image.
    """
    image = Image.open(img_path)

    #Resize image if it exceeds the maximum size
    if max(image.size) > max_size:
        scale  = max_size / float(max(image.size))
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size)

    #Convert image to tensor and normalize
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    # Add batch dimension for compatibility with model
    image = transform(image).unsqueeze(0)
    return image