import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

def predict_fracture(image_path, model_path="./pretrained_models/bone_fracture_model.pth"):
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    # If you saved the model state_dict within a dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        from torchvision import models
        import torch.nn as nn
        
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If the entire model object was saved
        model = checkpoint
    
    model = model.to(device)
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    class_names = ['fractured', 'not fractured']  # Ensure this order matches your training dataset
    result = {
        'prediction': class_names[predicted_class],
        'confidence': float(probabilities[predicted_class] * 100),
        'probabilities': {class_names[i]: float(probabilities[i] * 100) for i in range(len(class_names))}
    }
    
    # Visualize the results
    plt.figure(figsize=(10, 5))
    
    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')
    
    # Display the probabilities
    plt.subplot(1, 2, 2)
    bars = plt.bar(class_names, [result['probabilities'][c] for c in class_names])
    plt.title('Prediction Confidence')
    plt.ylabel('Confidence (%)')
    plt.ylim(0, 100)
    
    # Color the bars based on prediction
    for i, bar in enumerate(bars):
        bar.set_color('green' if i == predicted_class else 'red')
    
    for i, v in enumerate([result['probabilities'][c] for c in class_names]):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    
    # Save the visualization
    output_dir = "./predictions"
    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    plot_path = os.path.join(output_dir, f"{image_name}_prediction.png")
    plt.savefig(plot_path)
    
    print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f}%)")
    print(f"Visualization saved to {plot_path}")
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict bone fracture from X-ray image')
    parser.add_argument('image_path', type=str, help='Path to the X-ray image')
    parser.add_argument('--model', type=str, default='./pretrained_models/bone_fracture_model.pth', 
                        help='Path to the trained model')
    
    args = parser.parse_args()
    predict_fracture(args.image_path, args.model)
