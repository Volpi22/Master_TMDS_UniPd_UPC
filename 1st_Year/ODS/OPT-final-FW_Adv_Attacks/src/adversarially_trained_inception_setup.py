import torch
import timm

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from PIL import Image

# Define the model name for the adversarially trained Inception v3
class FlatImageFolder(Dataset):
    """Custom Dataset for a flat folder of images"""
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = sorted([os.path.join(root, f) for f in os.listdir(root) 
                               if os.path.isfile(os.path.join(root, f)) and f.lower().endswith(('png', 'jpg', 'jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Return a dummy label, as we don't have ground truth
        return image, -1 

def get_model(device):
    """
    Loads the pre-trained adversarially trained Inception ResNet v2 model from timm.
    """
    model = timm.create_model('inception_resnet_v2.tf_ens_adv_in1k', pretrained=True)
    model.to(device)
    model.eval()
    return model

def get_dataset(data_path='/Users/gianfranco/Desktop/ODS/val_images'):  # dataset can be downloaded from https://huggingface.co/datasets/ILSVRC/imagenet-1k/tree/main/data
    """
    Loads the validation image dataset from a flat directory.
    """
    model = timm.create_model('inception_resnet_v2.tf_ens_adv_in1k', pretrained=False)
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    
    dataset = FlatImageFolder(root=data_path, transform=preprocess)
    return dataset

def get_correctly_classified_images(model, device, dataset, num_images=100):
    """
    Finds images and assigns their predicted label as ground truth.
    """
    correctly_classified = []
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    pbar = tqdm(total=num_images, desc="Finding model-classified images")

    with torch.no_grad():
        for images, _ in data_loader:
            if len(correctly_classified) >= num_images:
                break

            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correctly_classified.append((images.cpu(), predicted.cpu()))
            pbar.update(1)
    
    pbar.close()
    print(f"Found {len(correctly_classified)} images to attack.")
    return correctly_classified
