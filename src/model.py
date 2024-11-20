import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import numpy as np
from src.preprocessing import load_image

#Define device: Using GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Defining model class for VGG-19
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        #Load the pre-trained VGG-19 model
        self.model = models.vgg19(pretrained=True).features

        #Freeze layers to prevent backpropagation
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)
    
def run_style_transfer(content_path, style_path, output_path, num_steps=300, content_weight=1, style_weight = 1e6):
    """
    Run the style transfer process with the given content and style images.
    Args:
    - content_path (str): path to the content image.
    - style_path (str): path to the style image.
    - output_path (str): path to save the output image.
    - num_steps (int): number of optimization steps.
    - content_weight (int): weight for the content loss.
    - style_weight (int): weight for the style loss.
    """

    #load content and style images
    ...
    



