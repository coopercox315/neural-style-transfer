import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import numpy as np
import argparse

#Define device: Using GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
#Defining model class for VGG-19
class ModifiedVGG19(nn.Module):
    def __init__(self, content_img, style_img, content_layers, style_layers):
        super(ModifiedVGG19, self).__init__()
        #Load the pre-trained VGG-19 model
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
        self.model = nn.Sequential()
        self.content_losses = []
        self.style_losses = []
        x_content = content_img.clone().to(device)
        x_style = style_img.clone().to(device)
        
        #Add layers and insert loss modules where required
        for name, layer in vgg._modules.items():
            #set inplace=False for ReLU layers so they work with the loss modules
            if isinstance(layer, nn.ReLU):
                layer.inplace = False

            self.model.add_module(name, layer)
            x_content = layer(x_content)
            x_style = layer(x_style)

            #Add content loss layer
            if name in content_layers:
                target = x_content.detach()
                content_loss = ContentLoss(target) #We will set the target later
                self.model.add_module(f"content_loss_{name}", content_loss)
                self.content_losses.append(content_loss)
            
            #Add style loss layer
            if name in style_layers:
                target = x_style.detach()
                style_loss = StyleLoss(target) #We will set the target later
                self.model.add_module(f"style_loss_{name}", style_loss)
                self.style_losses.append(style_loss)

            #Stop after the last relevant layer to save computation
            if len(self.content_losses) >= len(content_layers) and len(self.style_losses) >= len(style_layers):
                break

        #Freeze layers to prevent backpropagation
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)
    
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
    
#define function to save the generated image
def save_image(tensor, path):
    """
    Save a PyTorch tensor as an image file.
    Args:
    - tensor (Tensor): the input tensor.
    - path (str): the path to save the image file.
    """
    tensor = tensor.clone().detach().cpu().squeeze(0)
    unnormalize = T.Compose([
        T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
        T.Lambda(lambda x: x.clamp(0, 1))
    ])
    img = unnormalize(tensor).numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)

def run_style_transfer(content_path, style_path, output_path, max_size=512, num_steps=300, content_weight=1, style_weight = 1e6, progress_callback=None):
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
    content_img = load_image(content_path, max_size=max_size).to(device)
    style_img = load_image(style_path, max_size=max_size, shape=content_img.shape[-2:]).to(device)

    #Initialize the output image as the content image
    gen_img = content_img.clone().requires_grad_(True).to(device)

    #Define the layers for content and style losses as per Gatys et al. (2016)
    content_layers = ['21'] #conv4_2
    style_layers = ['0', '5', '10', '19', '28'] #'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'

    #Initialize the modified VGG-19 model using the defined content and style layers and images
    model = ModifiedVGG19(content_img, style_img, content_layers, style_layers).to(device)
    model.eval()

    #Define the optimizer
    optimizer = torch.optim.LBFGS([gen_img]) #using L-BFGS optimizer for optimization

    #mutable variable to keep track of the number of optimization steps
    step_counter = [0]

    #Optimization loop 
    while step_counter[0] <= num_steps:
        def closure():
            optimizer.zero_grad()
            model(gen_img) #pass the generated/output image through the model
            c_loss = sum([cl.loss for cl in model.content_losses])
            s_loss = sum([sl.loss for sl in model.style_losses])
            total_loss = (content_weight * c_loss) + (style_weight * s_loss)

            total_loss.backward()

            #Log progress
            step_counter[0] += 1
            if step_counter[0] % 50 == 0:
                print(f"Step [{step_counter[0]}/{num_steps}], Content Loss: {c_loss.item():.4f}, Style Loss: {s_loss.item():.6f}")
            return total_loss
        optimizer.step(closure)
        if progress_callback and step_counter[0] <= num_steps:
            progress = step_counter[0]/num_steps
            progress_callback(progress)

    #Save the generated image
    save_image(gen_img, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Neural Style Transfer")
    parser.add_argument("--content", type=str, required=True, help="Path to the content image.")
    parser.add_argument("--style", type=str, required=True, help="Path to the style image.")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save the output image.")
    parser.add_argument("--max_size", type=int, default=512, help="Maximum size for the content and style images.")
    parser.add_argument("--steps", type=int, default=300, help="Number of optimization steps.")
    parser.add_argument("--content_weight", type=int, default=1, help="Weight for the content loss.")
    parser.add_argument("--style_weight", type=int, default=1e6, help="Weight for the style loss.")

    args = parser.parse_args()

    #Run style transfer with the provided arguments
    run_style_transfer(args.content, args.style, args.output, args.max_size, args.steps, args.content_weight, args.style_weight)