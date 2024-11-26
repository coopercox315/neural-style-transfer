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
    
def gram_matrix(tensor):
    """
    Compute the Gram matrix of a given input tensor.
    Args:
    - input (Tensor): the input tensor.
    
    Returns:
    - Tensor: the Gram matrix of the input tensor.
    """
    #Get the batch size, number of channels, height, and width of the input tensor
    b, c, h, w = tensor.size()
    
    #Reshape the input tensor to be a 2D matrix
    features = tensor.view(b * c, h * w)
    
    #Compute the Gram matrix
    g_mat = torch.mm(features, features.t())
    
    #return the normalized Gram matrix
    return g_mat.div(b * c * h * w)
    
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach() #detach the target tensor to prevent gradient computation
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input
    
class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach() #detach the target tensor to prevent gradient computation
    def forward(self, input):
        g_mat = gram_matrix(input)
        self.loss = nn.functional.mse_loss(g_mat, self.target)
        return input
    
def run_style_transfer(content_path, style_path, output_path, num_steps=300, content_weight=1, style_weight = 1e6):
    """
    Run the style transfer process with the given content and style images.
    The ratio of content_weight to style_weight determines the balance between
    the two losses and affects how much the output image will be stylized.
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
    



