# Fine Art Artist Classification: ResNet, DenseNet & EfficientNet from Scratch

This deep learning project classifies fine art paintings by identifying the artist among 50 classical and modern masters. Instead of relying on pre-trained models or high-level wrappers, this project builds core Convolutional Neural Network (CNN) architectures—specifically **ResNet**, **DenseNet**, and **EfficientNet**—entirely from scratch using PyTorch.

## 🚀 Key Architectures

- **ResNet18:** A custom implementation of the 18-layer Residual Network using standard `ResNetBlock` units with skip connections.
- **ResNet50:** A deeper implementation utilizing the `Bottleneck` block architecture (expansion factor of 4) to increase network depth without significantly increasing computational cost.
- **DenseNet101:** A custom implementation of the Dense Convolutional Network where each layer receives feature maps from all preceding layers, improving gradient flow and feature reuse.
- **EfficientNet:** Built from the ground up utilizing Mobile Inverted Bottleneck Convolution (MBConv) blocks, Batch Normalization, and SiLU activations to optimize both network efficiency and accuracy.

## 🖼️ Custom Dataset & Balancing
This project uses a curated dataset of artworks from **50 distinct artists** (e.g., Leonardo da Vinci, Vincent van Gogh, Frida Kahlo). 
- **Automatic Balancing:** A custom script balances the dataset prior to training. It uses heavy augmentation to upsample underrepresented artists and caps highly represented ones to ensure a uniform distribution of exactly 300 images per artist.
- **Augmentations:** The pipeline uses `torchvision.transforms` for Random Rotations, Horizontal Flips, Resized Crops, Color Jittering, and Gaussian Blurs to generalize artistic styles.

> **Dataset Access:** Due to file size limits, the raw images are not hosted in this repository. If you would like access to the custom 50-artist dataset, please reach out to me via email at: [sayfitdinovsarvar12@gmail.com](mailto:sayfitdinovsarvar12@gmail.com)

## 🛠️ Technology Stack
- **Python 3**
- **PyTorch & Torchvision** (Model architectures and image transformations)
- **Weights & Biases (`wandb`)** (Real-time tracking of training loss and metrics)
- **Pandas / NumPy** (Data manipulation)

## 📂 File Structure
- `ArtworksArtistClassification.ipynb`: The main Google Colab notebook containing the data processing pipeline, the custom PyTorch classes for `ResNet18`, `ResNet50`, `DenseNet`, and `EfficientNet`, and the training loops.

## ⚙️ How to Run
1. Clone this repository or download the `.ipynb` file.
2. Open the notebook in Google Colab.
3. Upload your dataset to Google Drive and update the path in the `google.colab.drive.mount('/content/drive')` cell.
4. Obtain an API key from [Weights & Biases](https://wandb.ai/) for experiment tracking.
5. Instantiate your desired model class (ResNet, DenseNet, or EfficientNet) and run the cells to begin training.
