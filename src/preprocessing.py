import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#function to load and preprocess an image
def load_image(img_path, max_size=512, shape=None):
    """
    Load and preprocess an image from a given path.
    Args:
    - img_path (str): path to the image file.
    - max_size (int): the maximum size for resizing the image.
    - shape (tuple): the shape to resize the image to. Input content image shape when loading style image.
    
    Returns:
    - Tensor: a tensor representing the preprocessed image.
    """
    image = Image.open(img_path)

    #Resize image if it exceeds the maximum size and preserve aspect ratio of content image
    if shape is None:
        width, height = image.size
        max_dim = max(width, height)
        if max_dim > max_size:
            scale = max_size / float(max_dim)
            new_width = int(width * scale)
            new_height = int(height * scale)
            new_size = (new_height, new_width)  #T.Resize expects (H, W)
        else:
            new_size = (height, width)  #(H, W)
    else:
        #shape is already (H, W)
        new_size = shape

    #Convert image to tensor and normalize
    transform = T.Compose([
        T.Resize(new_size, T.InterpolationMode.BICUBIC),
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

#function to display content and style images
def display_images(content, style):
    """
    Display the content and style images.
    Args:
    - content (Tensor): the content image tensor.
    - style (Tensor): the style image tensor.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(to_numpy(content))
    ax2.imshow(to_numpy(style))
    plt.show()

# content = load_image("river.jpg")
# style = load_image("marg.jpg", shape=content.shape[-2:])
# display_images(content, style)