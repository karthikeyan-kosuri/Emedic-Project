import os
import kaggle
from external_trainer import train_model

# Set your Kaggle credentials directory
os.environ['KAGGLE_CONFIG_DIR'] = r"E:\Workspace\.kaggle"

def download_dataset_if_needed(data_path, kaggle_dataset):
    # Check if a 'train' folder exists; if not, assume dataset is unsplit
    if not os.path.isdir(os.path.join(data_path, 'train')):
        print("Dataset not found locally. Downloading from Kaggle...")
        print(f"Dataset URL: https://www.kaggle.com/datasets/{kaggle_dataset}")
        try:
            os.makedirs(data_path, exist_ok=True)
            kaggle.api.dataset_download_files(
                dataset=kaggle_dataset,
                path=data_path,
                unzip=True
            )
            print("✅ Dataset downloaded and unzipped successfully.")
        except Exception as e:
            print("❌ Error downloading dataset:", e)
            exit(1)
    else:
        print("Dataset found locally. Skipping download.")

if __name__ == "__main__":
    print("✅ Starting bone fracture classification training...")

    # Path to your dataset folder after unzipping
    data_path = "E:/Data/dataset/Bone_Fracture_Binary/Bone_Fracture_Binary_Classification"
    
    # Kaggle dataset slug
    kaggle_dataset = "bmadushanirodrigo/fracture-multi-region-x-ray-data"

    download_dataset_if_needed(data_path, kaggle_dataset)
    
    # Directory for saving models
    save_dir = "./pretrained_models"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "bone_fracture_model.pth")
    
    # To quickly simulate training, enable simulation mode by setting simulate=True
    model, test_accuracy = train_model(
        data_dir=data_path,
        model_save_path=model_save_path,
        epochs=5,
        batch_size=32,
        simulate=True  # Enable simulation mode for faster testing
    )
    
    print(f"✅ Training complete! Final test accuracy: {test_accuracy:.2f}%")
    print(f"✅ Model saved to: {model_save_path}")
#
# The script above is a Python script that trains a deep learning model to classify bone fractures in X-ray images.d
