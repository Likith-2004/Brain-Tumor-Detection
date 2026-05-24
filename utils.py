import torch
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image

def generate_gradcam(model, image_path, target_class, output_dir):
    model.eval()
    final_conv = model.layer4[-1]  # Last conv layer in ResNet18

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks
    h1 = final_conv.register_forward_hook(forward_hook)
    h2 = final_conv.register_backward_hook(backward_hook)

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    input_tensor.requires_grad = True

    # Forward pass
    output = model(input_tensor)
    model.zero_grad()

    # Backward pass
    class_score = output[0, target_class]
    class_score.backward()

    # Get hooks
    grads_val = gradients[0].squeeze().detach().numpy()
    activation = activations[0].squeeze().detach().numpy()

    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(activation.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activation[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max()

    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    orig_img = cv2.imread(image_path)
    orig_img = cv2.resize(orig_img, (224, 224))
    overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)

    # Save Grad-CAM
    filename = os.path.basename(image_path).split('.')[0] + "_gradcam.jpg"
    gradcam_path = os.path.join(output_dir, filename)
    cv2.imwrite(gradcam_path, overlay)

    # Cleanup hooks
    h1.remove()
    h2.remove()

    return gradcam_path