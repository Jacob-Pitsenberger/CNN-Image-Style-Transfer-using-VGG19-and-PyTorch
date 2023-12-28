"""
Author: Jacob Pitsenberger
Date: 12-28-23
Module: style_transfer.py

This module implements neural style transfer using PyTorch and VGG19. It enables the artistic transformation
of images by combining the content of one image with the style of another, leveraging deep learning techniques.

Key functionalities:
- Image loading and transformation using PyTorch and the VGG19 model.
- Feature extraction from different layers of the VGG19 model.
- Gram matrix calculation for capturing style representation.
- Content and style loss functions for iteratively updating a target image during style transfer.

Functions:
- load_image(img_path, max_size=400, shape=None): Load and transform an image.
- im_convert(tensor): Convert a PyTorch tensor to a NumPy image.
- display_content_and_style_images(content, style): Display the content and style images side-by-side.
- print_vgg_layers(vgg): Print the layers of the VGG model.
- get_features(image, model, layers=None): Extract features from a model.
- gram_matrix(tensor): Calculate the Gram Matrix of a given tensor.
- display_target_image(content, target): Display the target image along with the content image.
- image_style_transfer(vgg, content, style, device): Perform style transfer on the content image using the style image.
- main(): Main function for executing the PyTorch Style Transfer project.
"""

# import resources
import os
from PIL import Image  # Import the Image class from the PIL library for image processing
from io import BytesIO  # Import BytesIO for handling input/output streams
import matplotlib.pyplot as plt  # Import pyplot for plotting
import numpy as np  # Import NumPy for numerical computing

import torch  # Import PyTorch
import torch.optim as optim  # Import the optimization module from PyTorch
import requests  # Import the requests library for making HTTP requests
from torchvision import transforms, models  # Import transforms and models from torchvision for computer vision
from typing import Optional, Tuple  # Import type hints


def load_image(img_path: str, max_size: int = 400, shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Load and transform an image."""
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')  # Open and convert the image to RGB format
    else:
        image = Image.open(img_path).convert('RGB')  # Open and convert the image to RGB format

    size = max_size if max(image.size) > max_size else max(image.size)
    size = shape if shape is not None else size

    in_transform = transforms.Compose([
        transforms.Resize(size),  # Resize the image
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # Apply normalization

    image = in_transform(image)[:3, :, :].unsqueeze(0)  # Normalize, discard alpha channel, and add batch dimension
    return image


def im_convert(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a NumPy image."""
    image = tensor.to("cpu").clone().detach().numpy().squeeze().transpose(1, 2, 0)  # Convert tensor to NumPy array
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))  # Reverse normalization
    image = image.clip(0, 1)  # Clip values to the valid range [0, 1]
    return image


def display_content_and_style_images(content: torch.Tensor, style: torch.Tensor) -> None:
    """Display the content and style images side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))  # Create a subplot with two axes
    ax1.imshow(im_convert(content))  # Display the content image
    ax2.imshow(im_convert(style))  # Display the style image
    plt.show()


def print_vgg_layers(vgg: torch.nn.Module) -> None:
    """Print the layers of the VGG model."""
    print(vgg)  # Print the VGG model


def get_features(image: torch.Tensor, model: torch.nn.Module, layers: Optional[dict] = None) -> dict:
    """Extract features from a model."""
    if layers is None:
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x  # Store the features for specified layers
    return features


def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """Calculate the Gram Matrix of a given tensor."""
    b, d, h, w = tensor.size()
    tensor = tensor.view(b * d, h * w)  # Flatten the tensor
    gram = torch.mm(tensor, tensor.t())  # Calculate the Gram Matrix
    return gram


def display_target_image(content: torch.Tensor, target: torch.Tensor) -> None:
    """Display the target image along with the content image side-by-side and save the target image."""
    output_dir = "images"
    filename = "final_comparison_2000.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))  # Create a subplot with two axes
    ax1.imshow(im_convert(content))  # Display the content image
    ax2.imshow(im_convert(target))  # Display the target image

    # Save the target image to the output directory
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.show()


def image_style_transfer(vgg: torch.nn.Module, content: torch.Tensor, style: torch.Tensor,
                         device: torch.device) -> None:
    """Perform style transfer on the content image using the style image."""
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    target = content.clone().requires_grad_(True).to(device)  # Clone content image for stylization
    style_weights = {'conv1_1': 1., 'conv2_1': 0.75, 'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}
    content_weight = 1  # Content loss weight (alpha)
    style_weight = 1e6  # Style loss weight (beta)
    show_every = 400  # Show intermediate results every 400 iterations
    optimizer = optim.Adam([target], lr=0.003)  # Adam optimizer for target image optimization
    steps = 2000  # Number of optimization steps

    for ii in range(1, steps + 1):
        target_features = get_features(target, vgg)
        content_loss = torch.mean(
            (target_features['conv4_2'] - content_features['conv4_2']) ** 2)  # Calculate content loss

        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (d * h * w)  # Accumulate style loss for each layer
            print(f"For layer: {layer} loss is: {style_loss}")

        total_loss = content_weight * content_loss + style_weight * style_loss  # Calculate total loss
        print(f"for ii: {ii} total_loss is: {total_loss}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if ii % show_every == 0:
            print('Total loss: ', total_loss.item())
            plt.imshow(im_convert(target))
            plt.show()

    display_target_image(content, target)


def main() -> None:
    """Main function for executing the PyTorch Style Transfer project."""
    vgg = models.vgg19(weights='VGG19_Weights.DEFAULT').features
    for param in vgg.parameters():
        param.requires_grad_(False)  # Freeze all VGG parameters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg.to(device)

    content = load_image('images/me.png').to(device)
    style = load_image('images/vanGogh.png', shape=content.shape[-2:]).to(device)

    image_style_transfer(vgg, content, style, device)


if __name__ == "__main__":
    main()
