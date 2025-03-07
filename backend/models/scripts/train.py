from external_trainer import train_model
import os

if __name__ == "__main__":
    print("✅ Starting bone fracture classification training...")
    
    # Update this path to your dataset location
    data_path = "R:/Data/dataset/Bone_Fracture_Binary"
    
    # Create directory for saving models
    save_dir = "./pretrained_models"
    os.makedirs(save_dir, exist_ok=True)
    
    model_save_path = os.path.join(save_dir, "bone_fracture_model.pth")
    
    # Train with more epochs for better convergence
    model, test_accuracy = train_model(
        data_dir=data_path,
        model_save_path=model_save_path,
        epochs=15,
        batch_size=16
    )
    
    print(f"✅ Training complete! Final test accuracy: {test_accuracy:.2f}%")