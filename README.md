# neural-style-transfer
Neural Style Tranfer (NST) is a deep learning technique that effectively merges the content of one image with the artistic style of another, creating a new stylized image. It leverages convolutional neural networks (CNNs) to extract and manipulate features from the style image and transfer them across to the content image, allowing users to take any image and turn it into a newly generated art piece.

This repo contains a PyTorch implementation of 'Image Style Transfer' as discusssed in the original paper by [Gatys et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) The PyTorch implementation by [Alexis Jacq](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) was also referenced during the creation of this version.

## How it works
Neural Style Transfer relies on CNNs to seperate and recombine the **content** and **style** of images, creating visually impressive results. This project implements NST using a modified VGG19 model, organized in a class-based structure for greater flexibility and modularity. The process itself involves the following key steps:
1. **Image loading and preprocessing**: before the NST process begins, the input images (content and style) must be prepared for the model. This involves resizing, normalizing and converting them into a format that is compatible with the model (tensor).
   - **Image Resizing**:
     - The input images are resized to a maximum size specified by the user (e.g., 512). This ensures consistency and reduces the computational load for large images.
     - The style image is further adjusted to match the dimensions of the content image to ensure the feature extraction process is successful.
   - **Conversion to Tensors**:
     - The resized images are converted to PyTorch tensors, which are the data format required by the neural network for processing.
     - Tensors allow the images to be fed into the model as numerical arrays for feature extraction.
   - **Normalization**:
     - Images are normalized to match the input distribution expected by the pretrained VGG19 model.
     - Pixel values are scaled to the range `[0, 1]`.
     - Mean and standard deviation normalization is applied to center the pixel values around zero. 
3. **Feature Extraction with a Pretrained CNN**: A pretrained CNN, such as VGG19, is used to extract visual features from images. These features are captured at various layers of the network:
   - **Shallow Layers**: Used for extracting low-level features such as edges, textures, and colors.
   - **Deeper Layers**: Used for extracting high-level features such as shapes and the overall structure of objects in the image.

   Below is the model architecture used in this project, highlighting the layers chosen for content and style extraction:
   ![CNN](https://github.com/user-attachments/assets/e3260255-c648-453a-8835-658c4fa621ab)

   In my implementation, the **ModifiedVGG19** class wraps the original VGG19 model and inserts **custom layers** (e.g., `ContentLoss` and `StyleLoss`) at specific points in the architecture.
5. **Defining Content and Style Representations**

   **Content Representation**
     - The **content** of an image is extracted from a deeper layer of the CNN (e.g., `conv4_2` in VGG19).
     - This layer preserves the structural layout and major elements of the content image.
     - The `ContentLoss` layer computes the Mean Squared Error (MSE) between the feature maps of the generated image and the content image, ensuring the output retains the original image structure.

   **Style Representation**
    - The **style** of an image is captured using **Gram matrices**, which represent the correlation between different feature maps at each layer.
    - These correlations encode patterns, textures, and artistic styles, such as brush strokes or color palettes.
    - The `StyleLoss` layer compares the Gram matrices of the generated image and the style image, using MSE to match the style features.
  
   The diagram below shows **feature maps** extracted at different layers of the VGG19 network when processing the two images from the previous diagram (**Tiger** as the content image, **Starry Night** as the style image):
   ![Feature_maps](https://github.com/user-attachments/assets/6f577bab-510f-4857-9090-d0c7dd15310a)

7. **Modifying the VGG19 Model**: Instead of a simple function-based approach like other NST repos, my approach uses a **class-based design** for modularity and clarity. The `ModifiedVGG19` class:
   - Wraps the pretrained VGG19 model.
   - Iteratively inserts custom `ContentLoss` and `StyleLoss` layers at specified points during the network.
   - After adding the loss layers, the model stops further processing to save computation.

   **Why use classes?**
   
   The class structure includes benefits such as:
   - Better layer management.
   - Being able to reuse losses during optimization.
   - Modular design, keeping NST pipeline (loading images, running style transfer, saving output) clean and seperate from the VGG19 modifications
   - Readability, as all logic related to modifying the model and inserting layers is within this one class.
   - Scalability, architectures can be switched within the class without having to disrupt the overall NST pipeline
  
   The setup used for this implementation placed content and style layers at the following points:
   - **Content Layers**: `conv4_2` (layer `21` in VGG19 model)
   - **Style Layers**: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1` and `conv5_1` (layers `0`, `5`, `10`, `19` and `28`)
8. **Optimizing the Output Image**: The NST process treats image generation as an optimization problem:
    1. **Initialization**:
       - The generated image starts as a copy of the content image
    2. **Loss Function**:
       - **Content Loss**: Preserves structural details from the content image
       - **Style Loss**: Matches the textures and patterns of the style image using gram matrices.
       - **Total Loss**: Weighted sum of content and style losses:
         <img src ="https://github.com/user-attachments/assets/f00398e1-9318-4157-8c59-a83d89f4998f" width=600px>

         - $\alpha$: Content weight (e.g., 1).
         - $\beta$: Style weight (e.g., 1e6 or 1e7)
    3. **Gradient Descent**:
       - Using the **L-BFGS optimizer**, the generated image is updated iteratively to minimize the total loss.
       - Custom `closure()` function ensures the losses are recalculated and backpropagated for each step.
           
## Examples
The following examples showcase output images generated using only the code implemented in this repo. Each result combines the content of one image with the artistic style of another, highlighting the power of NST as executed by this project.

<div align="center">
   <img src="https://github.com/coopercox315/neural-style-transfer/blob/main/examples/content/tiger.jpg?raw=true" width=33%>
   <img src="https://github.com/coopercox315/neural-style-transfer/blob/main/examples/style/starry.jpg?raw=true" width=33%>
   <img src="https://github.com/coopercox315/neural-style-transfer/blob/main/examples/output/tiger_starry.jpg?raw=true" width=33%>
</div>
<div align="center">
   <img src="https://github.com/coopercox315/neural-style-transfer/blob/main/examples/content/cityscape.jpeg?raw=true" width=33%>
   <img src="https://github.com/coopercox315/neural-style-transfer/blob/main/examples/style/compositionvii.jpg?raw=true" width=33%>
   <img src="https://github.com/coopercox315/neural-style-transfer/blob/main/examples/output/cityscape_composition.jpg?raw=true" width=33%>
</div>
<div align="center">
   <img src="https://github.com/coopercox315/neural-style-transfer/blob/main/examples/content/neckarfront.jpg?raw=true" width=33%>
   <img src="https://github.com/coopercox315/neural-style-transfer/blob/main/examples/style/shipwreck.jpg?raw=true" width=33%>
   <img src="https://github.com/coopercox315/neural-style-transfer/blob/main/examples/output/neckarfront_shipwreck.jpg?raw=true" width=33%>
</div>
<div align="center">
   <img src="https://github.com/coopercox315/neural-style-transfer/blob/main/examples/content/bwportrait.jpg?raw=true" width=33%>
   <img src="https://github.com/coopercox315/neural-style-transfer/blob/main/examples/style/abstractswirls.jpeg?raw=true" width=33%>
   <img src="https://github.com/coopercox315/neural-style-transfer/blob/main/examples/output/bwportrait_swirls.jpg?raw=true" width=33%>
</div>
