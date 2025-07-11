import os
import torch
from torchvision import datasets, transforms

def get_correctly_classified_images(model, device, dataset_name, num_images, data_root='./data', model_name='resnet50'):
    """
    Finds or loads a specified number 
    of correctly classified images for a given model and dataset.

    If a saved file of images exists, it loads them. Otherwise, it searches the dataset,
    saves the found images to a file, and then returns them.

    Args:
        model (torch.nn.Module): The pretrained model to use for classification.
        device (torch.device): The device to run the model and tensors on.
        dataset_name (str): The name of the dataset (e.g., 'cifar10').
        num_images (int): The number of correctly classified images to find.
        data_root (str): The root directory where the dataset is stored.
        model_name (str): The name of the model, used for the save file.

    Returns:
        list: A list of tuples, where each tuple is (image_tensor, label_tensor).
    """
    save_path = os.path.join(data_root, f'correctly_classified_{dataset_name}_{model_name}.pt')

    if os.path.exists(save_path):
        print(f"Loading correctly classified images from {save_path}...")
        correctly_classified_images_cpu = torch.load(save_path)
        correctly_classified_images = [(img.to(device), lbl.to(device)) for img, lbl in correctly_classified_images_cpu]
        print(f"Loaded {len(correctly_classified_images)} images.")
        return correctly_classified_images

    print(f"Searching for {num_images} correctly classified images from {dataset_name.upper()}...")

    if dataset_name.lower() == 'cifar10':
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=preprocess)
        
        cifar10_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        cifar_to_imagenet = {
            'airplane': 404, 'automobile': 436, 'bird': 12, 'cat': 281, 'deer': 354,
            'dog': 207, 'frog': 30, 'horse': 339, 'ship': 780, 'truck': 867
        }
        class_idx_to_target_idx = {i: cifar_to_imagenet[name] for i, name in enumerate(cifar10_class_names)}
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
    
    found_images = []
    model.eval()
    model.to(device)

    for image, label in test_loader:
        if len(found_images) >= num_images:
            break
        
        image, label = image.to(device), label.to(device)
        
        with torch.no_grad():
            output = model(image)
            _, pred = torch.max(output, 1)

        target_idx = class_idx_to_target_idx[label.item()]

        if pred.item() == target_idx:
            found_images.append((image.squeeze(0), torch.tensor([target_idx], device=device)))

    print(f"Found {len(found_images)} correctly classified images.")
    
    if found_images:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        found_images_cpu = [(img.cpu(), lbl.cpu()) for img, lbl in found_images]
        print(f"Saving images to {save_path}...")
        torch.save(found_images_cpu, save_path)
        print("Images saved.")

    return found_images