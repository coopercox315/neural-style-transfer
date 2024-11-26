import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

    #Convert image to tensor and normalize
    transform = T.Compose([
        T.Resize(new_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    # Add batch dimension for compatibility with model
    image = transform(image).unsqueeze(0)
    return image

#function to unnormalize and image and convert to numpy image for display
def to_numpy(tensor):
    """
    Convert a PyTorch tensor to a numpy array for display.
    Args:
    - tensor (Tensor): a PyTorch tensor.
    
    Returns:
    - numpy.ndarray: the corresponding numpy array.
    """
    mean, std  = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = image.permute(1, 2, 0)
    image = (image * std) + mean
    image = np.clip(image, 0, 1)

    return image