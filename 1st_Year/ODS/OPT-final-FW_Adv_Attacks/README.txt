=======================================================================
    G04 ADVERSARIAL ATTACKS - OPTIMIZATION FOR DATA SCIENCE PROJECT
=======================================================================

Group Members: Elia Carta, Jianan E, Gianfranco Mauro, Davide Volpi
Course: Optimization for Data Science - University of Padova


FOLDER STRUCTURE
----------------
OPT-final-FW_Adv_Attacks/
├── README.txt                           # This file
├── G04AdversarialAttacks_Report.pdf     # Project report
└── src/                                 # Source code directory
    ├── frank_wolfe.ipynb                # Classical Frank-Wolfe algorithm
    ├── away_step_fw.ipynb               # Away-step Frank-Wolfe variant
    ├── pairwise_fw.ipynb                # Pairwise Frank-Wolfe variant
    ├── projected_gradient.ipynb         # Projected Gradient method
    ├── frank_wolfe_wb.ipynb             # FW-based white-box attacks
    ├── away_step_fw_wb.ipynb            # Away-step FW white-box attacks
    ├── pairwise_fw_wb.ipynb             # Pairwise FW white-box attacks
    ├── projected_gradient_wb.ipynb      # Projected Gradient white-box attacks
    ├── cifar_setup.py                   # CIFAR-10 dataset setup utilities
    ├── real_vs_fake_setup.py            # AI vs Real image setup utilities
    └── adversarially_trained_inception_setup.py  # Adversarial model setup
    └── data/                            # Datasets directory
        ├── correctly_classified_cifar10_resnet50.pt                                   # Correctly classified images for the CIFAR-10 dataset
        ├── correctly_classified_real_vs_fake_dima806_ai_vs_real_image_detection.pt    # Correctly classified images for the real vs fake dataset
        ├── correctly_classified_val_images_adv_inception_resnet_v2.pt                 # Correctly classified images for the adversarially trained model


REQUIRED LIBRARIES
------------------
Core Dependencies:
- torch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- transformers >= 4.20.0
- PIL (Pillow) >= 8.3.0


Optional:
- CUDA-compatible PyTorch for GPU acceleration
- MPS-compatible PyTorch for Apple Silicon acceleration

DATASETS
--------
1. CIFAR-10:
   - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
   - Images resized to 224x224 for ResNet compatibility
   - Automatically downloaded via torchvision

2. Real vs Fake Images:
   - Binary classification: AI-generated vs real images
   - Uses dima806/ai_vs_real_image_detection model from HuggingFace
   - Custom dataset setup for adversarial attack evaluation

3. ImageNet1k:
   - 1000 classes of objects
   - Standard validation set used with adversarially trained Inception v3
   - Used to evaluate robustness of adversarial attacks against defensive training


MODELS TESTED
-------------
1. ResNet-50 (pretrained on ImageNet, adapted for CIFAR-10)
2. AI vs Real Image Detection Model (dima806/ai_vs_real_image_detection)
3. Adversarially Trained Inception v3 (robustness evaluation)