import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_loss = model_output[0, target_class]
        class_loss.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Weight activations by gradients (global average pooling)
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        cam = (weights * activations).sum(dim=0)  # [H, W]
        
        # Apply ReLU (only positive influences)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.cpu().numpy()

def apply_colormap_on_image(org_img, activation_map, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on original image"""
    # Resize activation map to match image size
    heatmap = cv2.resize(activation_map, (org_img.shape[1], org_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # Overlay heatmap on image
    superimposed_img = cv2.addWeighted(org_img, 0.6, heatmap, 0.4, 0)
    
    return superimposed_img

def load_model(model_path):
    """Load trained model"""
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, 2)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def predict_and_visualize(image_path, model, output_path):
    """Predict and generate Grad-CAM visualization"""
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load original image for visualization
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (224, 224))
    
    # Transform for model
    pil_img = Image.open(image_path).convert('RGB')
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    input_tensor.requires_grad = True
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    # Initialize Grad-CAM
    target_layer = model.features[-1]  # Last convolutional layer
    grad_cam = GradCAM(model, target_layer)
    
    # Generate CAM for predicted class
    cam = grad_cam.generate_cam(input_tensor, predicted_class)
    
    # Apply colormap
    visualization = apply_colormap_on_image(original_img, cam)
    
    # Create figure with results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(visualization)
    label = "REAL" if predicted_class == 0 else "FAKE"
    axes[2].set_title(f'Prediction: {label} ({confidence*100:.1f}% confidence)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Prediction: {label} ({confidence*100:.1f}% confidence)")
    print(f"Visualization saved to: {output_path}")
    
    return label, confidence

def batch_visualize(image_dir, model, output_dir, num_samples=10):
    """Generate visualizations for multiple images"""
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')][:num_samples]
    
    print(f"\nGenerating Grad-CAM visualizations for {len(image_files)} images...")
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        output_path = os.path.join(output_dir, f"gradcam_{img_file}")
        
        try:
            predict_and_visualize(img_path, model, output_path)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    print(f"\n✅ All visualizations saved to: {output_dir}")

if __name__ == "__main__":
    # Load trained model (try fixed first, fallback to alternative)
    MODEL_PATHS = ['best_model_fixed.pth', 'best_model.pth']
    model = None
    for mp in MODEL_PATHS:
        try:
            model = load_model(mp)
            print(f"Model loaded successfully from: {mp}")
            break
        except Exception as e:
            print(f"Failed loading {mp}: {e}")
    if model is None:
        raise SystemExit("No valid model could be loaded. Check the .pth files in the repo.")
    
    # Create output directory
    OUTPUT_DIR = './gradcam_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Test on some real faces
    print("\n" + "="*200)
    print("Generating Grad-CAM for REAL faces")
    print("="*200)
    batch_visualize('./processed_faces/real', model, OUTPUT_DIR + '/real', num_samples=5)
    
    # Test on some fake faces
    print("\n" + "="*200)
    print("Generating Grad-CAM for FAKE faces")
    print("="*200)
    batch_visualize('./processed_faces/fake', model, OUTPUT_DIR + '/fake', num_samples=5)
    
    print("\n" + "="*200)
    print("✅ Done! Check gradcam_results/ folder")
    print("="*50)