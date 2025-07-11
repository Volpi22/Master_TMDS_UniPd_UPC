# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-classification", model="dima806/ai_vs_real_image_detection")

# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/ai_vs_real_image_detection")

import os
import torch
from PIL import Image
from torchvision import datasets
import random

def get_correctly_classified_images_real_vs_fake(model, processor, device, dataset_path, num_images, data_root='./data', model_name='dima806_ai_vs_real_image_detection'):
    """
    Finds or loads a specified number of correctly classified images for the real-vs-fake model.

    If a saved file of images exists, it loads them. Otherwise, it searches the dataset,
    saves the found images to a file, and then returns them.

    Args:
        model (torch.nn.Module): The pretrained model to use for classification.
        processor (transformers.AutoImageProcessor): The processor for the model.
        device (torch.device): The device to run the model and tensors on.
        dataset_path (str): The path to the root of the test dataset.
        num_images (int): The number of correctly classified images to find.
        data_root (str): The root directory to save the found images file.
        model_name (str): The name of the model, used for the save file.

    Returns:
        list: A list of tuples, where each tuple is (image_tensor, label_tensor).
    """
    save_path = os.path.join(data_root, f'correctly_classified_real_vs_fake_{model_name}.pt')

    if os.path.exists(save_path):
        print(f"Loading correctly classified images from {save_path}...")
        correctly_classified_images_cpu = torch.load(save_path)
        correctly_classified_images = [(img.to(device), lbl.to(device)) for img, lbl in correctly_classified_images_cpu]
        print(f"Loaded {len(correctly_classified_images)} images.")
        return correctly_classified_images

    print(f"Searching for {num_images} correctly classified images from {dataset_path}...")

    test_dataset = datasets.ImageFolder(root=dataset_path)
    model_label_map = {v: model.config.label2id[k] for k, v in test_dataset.class_to_idx.items()}

    found_images = []
    model.eval()
    model.to(device)

    image_files = test_dataset.samples
    random.shuffle(image_files)

    for image_path, label_idx in image_files:
        if len(found_images) >= num_images:
            break
        
        try:
            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping corrupted image {image_path}: {e}")
            continue
        
        inputs = processor(images=image_pil, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model(**inputs)
            logits = output.logits
            pred_label_idx = torch.argmax(logits, dim=1).item()

        target_label_idx = model_label_map[label_idx]

        if pred_label_idx == target_label_idx:
            image_tensor = inputs['pixel_values'].squeeze(0) 
            label_tensor = torch.tensor([target_label_idx], device=device)
            found_images.append((image_tensor, label_tensor))
            if len(found_images) % 100 == 0 or len(found_images) == num_images:
                print(f"Found {len(found_images)}/{num_images} images...")

    print(f"Found {len(found_images)} correctly classified images.")
    
    if found_images:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        found_images_cpu = [(img.cpu(), lbl.cpu()) for img, lbl in found_images]
        print(f"Saving images to {save_path}...")
        torch.save(found_images_cpu, save_path)
        print("Images saved.")

    return found_images