# AgriNet-Research

A comprehensive deep learning research project for agricultural computer vision, specializing in semantic segmentation and anomaly detection in aerial farm imagery using state-of-the-art neural network architectures.

## Overview

AgriNet-Research leverages modern deep learning techniques to analyze agricultural imagery captured by drones (DJI). The project focuses on:

- **Semantic Segmentation**: Identifying different crop types and field anomalies (e.g., weeds, disease, irrigation issues) in multi-spectral aerial images
- **Data Processing & Augmentation**: Advanced image preprocessing, normalization, and augmentation techniques
- **Model Training & Evaluation**: Comprehensive training pipelines with multiple architectures
- **Exploratory Data Analysis (EDA)**: Detailed statistical and visual analysis of agricultural datasets

## Project Structure

```text
agrinet/
├── best_models/              # Pre-trained model weights
│   ├── best_model_efficientnet-b2_dataset.pth
│   ├── best_model_efficientnet-b3_dataset.pth
│   ├── best_model_efficientnet-b5_dataset.pth
│   ├── best_model_mobilenet_v2_dataset.pth
│   └── best_model_resnet50_dataset.pth
├── datasets/
│   ├── Agriculture-Vision-2021/    # Primary dataset
│   │   ├── eda_and_normalization.ipynb          # EDA and normalization analysis
│   │   ├── work_with_dataset_and_train_model.ipynb  # Training pipeline
│   │   ├── create_mini_dataset.py               # Dataset creation utilities
│   │   ├── _dataset_processed_masks.csv         # Processed mask annotations
│   │   ├── _dataset_processed_aug_masks.csv     # Augmented mask annotations
│   │   └── _dataset/                            # Dataset directory structure
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   └── DJI_202507131523_004/       # Additional DJI dataset
│       ├── _dataset/
│       │   └── test/
│       └── _predictions_zscore/    # Z-score normalized predictions
└── training_results/          # Training analysis notebooks
    ├── results.ipynb
    ├── training_results_efficientnetB2.ipynb
    ├── training_results_efficientnetB3.ipynb
    ├── training_results_efficientnetB5.ipynb
    └── training_results_mobilenetV2.ipynb
```

## Supported Models

The project includes implementations and pre-trained weights for the following architectures:

- **EfficientNet-B2/B3/B5**: Efficient and scalable models for image classification and segmentation
- **ResNet50**: Classic deep residual network
- **MobileNet V2**: Lightweight architecture optimized for edge deployment

These models are adapted for semantic segmentation tasks using `segmentation-models-pytorch`.

## Datasets

### Agriculture-Vision-2021

Primary dataset with:

- Multi-spectral imagery from agricultural fields
- Pixel-level mask annotations for multiple anomaly classes
- Processed and augmented versions available
- Train/Val/Test splits

### DJI Drone Imagery

Real-world aerial captures from DJI drones with:

- Test set imagery
- Z-score normalized predictions for inference

## Installation

### Prerequisites

- Python 3.12+
- CUDA 12.4 (for GPU support) or CPU-only variant

### Setup with Poetry

```bash
# Clone the repository
git clone https://github.com/notbot479/AgriNet-Research.git
cd AgriNet-Research

# Create virtual environment and install dependencies
poetry install
```

### GPU Setup (CUDA 12.4 - RTX 40xx)

The default `pyproject.toml` is configured for GPU. Ensure CUDA 12.4 is installed:

```bash
poetry install
```

### CPU Setup

Modify `pyproject.toml` to use CPU wheels:

```toml
# Comment out GPU lines and uncomment CPU lines:
torch = {version = "2.5.1+cpu", source = "pytorch_cpu"}
torchvision = {version = "0.20.1+cpu", source = "pytorch_cpu"}
```

Then:

```bash
poetry install
```

## Dependencies

### Core Libraries

- **Deep Learning**: PyTorch 2.5.1, torchvision 0.20.1
- **Segmentation**: segmentation-models-pytorch 0.5.0
- **Data Processing**: NumPy 2, Pandas 2.2.3, scikit-image 0.26.0
- **Image Processing**: OpenCV 4.10.0, Albumentations 1.4.21
- **Visualization**: Matplotlib 3.9.2, Seaborn 0.13.2
- **ML Utilities**: scikit-learn 1.5.2, SciPy 1.14.1
- **Monitoring**: TensorBoard 2.18.0
- **Notebooks**: Jupyter 1.1.1

See `pyproject.toml` for complete dependency list.

## Usage

### 1. Exploratory Data Analysis

Open and run the EDA notebook:

```bash
jupyter notebook agrinet/datasets/Agriculture-Vision-2021/eda_and_normalization.ipynb
```

This notebook provides:

- Dataset statistics and distribution analysis
- Image normalization strategies
- Visualization of field anomalies
- Data augmentation preview

### 2. Training Models

Run the training pipeline:

```bash
jupyter notebook agrinet/datasets/Agriculture-Vision-2021/work_with_dataset_and_train_model.ipynb
```

This comprehensive notebook includes:

- Dataset loading and preprocessing
- Data augmentation strategies
- Model architecture setup
- Training loops with validation
- Loss tracking and metrics
- Model checkpointing

### 3. Create Mini Datasets

For quick prototyping:

```bash
cd agrinet/datasets/Agriculture-Vision-2021/
python create_mini_dataset.py
```

This creates a smaller dataset subset (default: 100 samples per folder) for faster iteration.

### 4. Analyze Training Results

Review model performance:

```bash
jupyter notebook agrinet/training_results/training_results_efficientnetB2.ipynb
# or for other models
jupyter notebook agrinet/training_results/training_results_efficientnetB3.ipynb
jupyter notebook agrinet/training_results/training_results_efficientnetB5.ipynb
jupyter notebook agrinet/training_results/training_results_mobilenetV2.ipynb
```

## Training Flow

The complete training pipeline is implemented in `work_with_dataset_and_train_model.ipynb`:

### 1. Dataset Preparation

- **Image Loading**: 4-channel RGBN imagery (Red, Green, Blue, Near-Infrared)
- **Train Dataset**: Combined original and augmented images (2 image directories)
- **Validation Dataset**: Separate RGB + NIR channels loaded and combined
- **Test Dataset**: US Agriculture-Vision dataset + optional Kazakhstan DJI dataset
- **Data Augmentation**: Applied to training set for improved model robustness

### 2. Multi-Label Segmentation Task

- **Classes**: 8 anomaly types (excludes `storm_damage`)
  - double_plant, drydown, endrow, nutrient_deficiency, waterway, water, planter_skip, weed_cluster
- **Mask Format**: Multi-label (overlapping anomalies allowed, not mutually exclusive)
- **Class Balancing**: Precomputed class weights to handle imbalanced datasets

### 3. Model Architecture

- **Base Model**: EfficientNet-B2/B3/B5, ResNet50, or MobileNet V2
- **Segmentation Head**: Adapted from `segmentation-models-pytorch`
- **Input**: 512×512 normalized RGBN images
- **Output**: Per-pixel predictions for each anomaly class

### 4. Training Configuration

- **Optimizer**: AdamW with weight decay (1e-4)
- **Learning Rate**: 1e-4 with Cosine Annealing scheduler
- **Batch Size**: 10 (GPU) or 2 (CPU)
- **Epochs**: 50 with early stopping (patience=10)
- **Mixed Precision**: Automatic mixed precision (AMP) for efficiency
- **Loss Function**: Combined BCE + Dice Loss (0.5 weight each)

### 5. Training Dynamics

- **Validation Frequency**: Every epoch
- **Metrics Tracked**:
  - Training loss
  - Validation loss
  - Mean Intersection over Union (mIoU) per class
- **Best Model Checkpoint**: Saved when validation mIoU improves
- **Resume Support**: Can resume training from checkpoint with full state preservation

### 6. Inference & Normalization

The pipeline supports multiple normalization strategies for domain adaptation:

- **SIMPLE**: Division by 255 (matches training data)
- **ZSCORE**: Linear mean/std normalization (for external datasets)
- **HISTOGRAM**: Non-parametric histogram matching to training reference

### 7. Prediction Output

- Per-class binary masks with configurable threshold (default: 0.5)
- NDVI (Normalized Difference Vegetation Index) calculation from RGBN
- Overlay visualization on original imagery
- Per-image detection summaries with pixel counts

## Workflow

1. **Data Preparation**: Run EDA notebook to understand dataset characteristics
2. **Data Preprocessing**: Apply normalization and augmentation
3. **Model Training**: Execute training pipeline with specified architecture
4. **Checkpoint Management**: Automatic best model saving and resume capability
5. **Results Analysis**: Review training metrics and model performance
6. **Inference**: Generate predictions with domain adaptation for new imagery

## Key Notebooks

| Notebook | Purpose |
| -------- | ------- |
| `eda_and_normalization.ipynb` | Dataset exploration, statistics, normalization analysis |
| `work_with_dataset_and_train_model.ipynb` | Complete training pipeline for semantic segmentation |
| `training_results_*.ipynb` | Model-specific performance analysis and visualizations |

## Model Outputs

Pre-trained model weights are stored in `agrinet/best_models/`:

- **File format**: PyTorch `.pth` checkpoints
- **Task**: Semantic segmentation
- **Input**: Normalized agricultural imagery
- **Output**: Pixel-level class predictions

## Configuration

Key parameters can be adjusted in the notebooks:

- Image size and normalization
- Augmentation strategies (flip, rotate, color jittering, etc.)
- Batch size and learning rate
- Model architecture and pretrained weights
- Training epochs and validation frequency

## Performance Metrics

The project tracks:

- Loss (Cross-Entropy, Dice Loss, etc.)
- Accuracy metrics
- IoU (Intersection over Union)
- F1 scores per class
- Confusion matrices

## Requirements

- RAM: 16GB+ recommended for full dataset training
- GPU VRAM: 8GB+ for comfortable training with batch size 32
- Disk: ~120GB for full dataset with augmentation

## References

- [Agriculture-Vision Challenge 2021](https://www.agriculture-vision.org/)
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Albumentations: Image Augmentation Library](https://albumentations.ai/)